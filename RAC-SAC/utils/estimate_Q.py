import numpy as np
import gym
import torch
from utils.others import NormalizedActions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_estimator:
    def __init__(self, config, agent, best_uncertain):
        self.config = config
        self.agent = agent
        self.best_uncertain = torch.FloatTensor(best_uncertain.reshape(1, -1)).to(device)

        self.eval_env = gym.make(self.config['env'])
        self.eval_env._max_episode_steps = 20000
        self.eval_env = NormalizedActions(self.eval_env)
        self.eval_env.seed(self.config['seed'] + 100)
        self.eval_env.action_space.seed(self.config['seed'])

        self.all_action = []
        num = int(100 / self.config['state_action_pairs'])
        for x in range(num, 100 + num, num):
            action_list = self.get_action_list(x, self.eval_env)
            self.all_action.append(action_list)

    def get_action_list(self, random_step, eval_env):
        action_list = []
        eval_env.seed(self.config['seed'] + 100)
        state, done = eval_env.reset(), False
        for x in range(random_step):
            action = self.agent.select_action(np.array(state),
                                              uncertain=self.best_uncertain)
            state, reward, done, _ = eval_env.step(action)
            action_list.append(action)
        return action_list

    def cal_Q_bias(self, action_list, MC_samples, max_mc_steps, eval_env):
        Q_mc = []
        Q_mean_list_all = []
        for x in range(MC_samples):
            eval_env.seed(self.config['seed'] + 100)
            state, done = eval_env.reset(), False
            for action in action_list[0:1]:
                last_state = state
                last_state_action = action
                state, reward, done, _ = eval_env.step(action)

            Q_mean, _ = self.agent.get_mean_std(torch.FloatTensor(last_state.reshape(1, -1)).to(device),
                                                torch.FloatTensor(last_state_action.reshape(1, -1)).to(device),
                                                self.best_uncertain)
            reward_list = []
            Q_mean_list = []
            total_reward = reward
            reward_list.append(reward)
            for y in range(max_mc_steps+200):
                state_ = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action, log_prob, _, _, _ = self.agent.get_action_log_prob(state_, self.best_uncertain)
                state, reward, done, _ = eval_env.step(action.cpu().data.numpy().flatten())
                Q_mean, _ = self.agent.get_mean_std(state_, action, self.best_uncertain)
                Q_mean_list.append(Q_mean.cpu().detach().numpy())
                logprob = float(log_prob)
                temperature = float(self.agent.temperature(self.best_uncertain))
                reward_term = (reward - logprob * temperature)
                total_reward += reward_term * self.agent.discount ** (y + 1)
                reward_list.append(reward_term)
                if done:
                    break
            vilid_index = len(reward_list) - 1
            total_reward_list = self.cal_reward_list(reward_list,self.agent.discount)
            total_reward_list = total_reward_list[0:vilid_index]
            Q_mean_list = Q_mean_list[0:vilid_index]
            # print(total_reward_list)
            Q_mc.extend(total_reward_list)
            Q_mean_list_all.extend(Q_mean_list)
        # print(Q_mean_list)
        # print("Q_MC")
        # print(Q_mc)
        bias = float(np.mean(Q_mean_list_all)) - float(np.mean(Q_mc))
        return bias, np.mean(Q_mc)
    
    def cal_reward_list(self, reward_list, discount):
        total_reward_list = []
        total_reward = 0
       # reverse the reward list
        reward_list.reverse()
        for reward in reward_list:
            total_reward = reward + discount * total_reward
            total_reward_list.append(total_reward)
        total_reward_list.reverse()
        return total_reward_list

    def cal_norm_bias(self):
        bias_list = []
        Q_mc_list = []
        for actions in self.all_action[0:1]:
            bias, Q_mc = self.cal_Q_bias(actions,
                                         MC_samples=self.config['MC_samples'],
                                         max_mc_steps=self.config['max_mc_steps'],
                                         eval_env=self.eval_env)
            bias_list.append(bias)
            Q_mc_list.append(Q_mc)
        norm_bias = np.array(bias_list) / abs(np.mean(Q_mc_list))
        norm_mean_bias = np.mean(norm_bias)
        norm_std_bias = np.std(norm_bias)
        abs_mean_bias = np.mean(bias_list)
        abs_std_bias = np.std(bias_list)
        abs_min_bias = np.min(bias_list)
        abs_max_bias = np.max(bias_list)
        Q_bias_dict = {
            'norm_mean_bias': norm_mean_bias,
            'norm_std_bias': norm_std_bias,
            'abs_mean_bias': abs_mean_bias,
            'abs_std_bias': abs_std_bias,
            'abs_min_bias': abs_min_bias,
            'abs_max_bias': abs_max_bias
        }
        return Q_bias_dict
