from collections import namedtuple
import torch
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQNAgent:
    def __init__(self, dqn, target_dqn):
        self.DQN = dqn
        self.target_dqn = target_dqn

    def update_param(self):
        self.target_dqn.load_state_dict(self.DQN.state_dict())

    def learn(self, batch_size):
        # if self.DQN.mm.__len__() < batch_size:
        #     return
        transitions = self.DQN.mm.sample(batch_size)
        batch = Transition(*zip(*transitions))
        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.DQN.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.DQN.device), batch.reward)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.DQN.device, dtype=torch.uint8).bool()

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.DQN.device)

        state_batch = torch.cat(batch.state).to(self.DQN.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        """# 用来选择
        state_batch shape: batch_size,8
        state_action_values shape: batch_size,1
        """
        state_action_values = self.DQN(state_batch).gather(1, action_batch)
        # next_state_values = torch.zeros(batch_size, device=self.DQN.device)
        # 1.带有target的DQN
        # next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).max(1)[0].detach()
        # 2.没有target
        # next_state_values[non_final_mask] = self.DQN(non_final_next_states).max(1)[0].detach()
        # print(next_state_values.shape)
        # 3.double dqn，这块注释了上面的next_state_values = torch.zeros(batch_size, device=self.DQN.device)
        next_state_actions = self.DQN(non_final_next_states).argmax(1, keepdim=True)
        next_state_values = torch.zeros(batch_size, device=self.DQN.device)
        next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).gather(1,
                                                                                          next_state_actions).squeeze(1)
        expected_state_action_values = (next_state_values * self.DQN.gamma) + reward_batch
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1).to(torch.float32))
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1).to(torch.float32)) / batch_size
        self.DQN.optimizer.zero_grad()
        loss.backward()
        # for param in self.DQN.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.DQN.optimizer.step()