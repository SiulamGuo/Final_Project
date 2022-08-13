import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter, functional as F
from torch.distributions import Normal

class actor(nn.Module):
    def __init__(self,
                 env_params,
                 embedding_dim=64,
                 activation_fnx=F.leaky_relu,
                 ):

        super(actor, self).__init__()
        # self.fc_embed = nn.Linear(env_params['obs'] + env_params['goal'], embedding_dim)
        self.fc_embed = nn.Linear(28, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.activation_fnx = activation_fnx

        self.attention = Attention(embedding_dim)
        self.AttentiveGraphPooling = AttentiveGraphPooling(embedding_dim)
        self.mlp = Mlp(env_params, embedding_dim)

    def forward(self, obs_input):
        x = self.fc_embed(obs_input)
        x1 = self.layer_norm(x)
        x2 = self.fc_qcm(x1)
        query, context, memory = x2.chunk(3, dim=-1)
        attention_result = self.attention(query, context, memory)

        result = x1 + attention_result
        output = self.activation_fnx(result)
        response_embeddings = self.layer_norm2(output)

        attention_result2 = self.AttentiveGraphPooling(response_embeddings)
        attention_result3 = self.AttentiveGraphPooling(attention_result2)
        action = self.mlp(attention_result3)

        return action

    def _preproc_inputs_new(observation):
        """
        original obs construction
            obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )

        achieved_goal = object_pos.copy()


        change to obs_new
            obs = np.concatenate(
            [
                grip_pos,
                gripper_state,
                grip_velp,
                gripper_vel,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ]
        )
        achieved_goal = np.concatenate([object_i_pos.copy(), grip_pos.copy()])
        """
        obs = observation['observation']

        grip_pos = obs[0:3]
        object_pos = obs[3:6]
        object_rel_pos = obs[6:9]
        gripper_state = obs[9:11]
        object_rot = obs[11:14]
        object_velp = obs[14:17]
        object_velr = obs[17:20]
        grip_velp = obs[20:23]
        gripper_vel = obs[23:25]

        obs_new = np.concatenate(
            [
                grip_pos,
                gripper_state,
                grip_velp,
                gripper_vel,
                object_pos,
                object_rel_pos,
                object_rot,
                object_velp,
                object_velr
            ]
        )

        ag_new = np.concatenate([object_pos.copy(), grip_pos.copy()])
        observation_ = observation.copy()
        observation_['observation'] = obs_new
        # observation_['achieved_goal'] = ag_new

        return observation_

    def numblock_preprocess(obs,
                            robot_dim=10,
                            object_dim=15,
                            goal_dim=3,
                            ):
        batch_size, environment_state_length = obs.size()
        nB = (environment_state_length - robot_dim) / (object_dim + goal_dim)
        nB = int(nB)

        robot_state_flat = obs.narrow(1, 0, robot_dim)
        flattened_objects = obs.narrow(1, robot_dim, object_dim * nB)
        batched_objects = flattened_objects.view(batch_size, nB, object_dim)
        flattened_goals = obs.narrow(1, robot_dim + nB * object_dim, nB * goal_dim)
        batched_goals = flattened_goals.view(batch_size, nB, goal_dim)
        batch_shared = robot_state_flat.unsqueeze(1).expand(-1, nB, -1)
        batch_objgoals = torch.cat((batched_objects, batched_goals), dim=-1)
        batched_combined_state = torch.cat((batch_shared, batch_objgoals), dim=-1)

        return batched_combined_state

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class Attention(nn.Module):
    def __init__(self,
                 embedding_dim,
                 activation_fnx=F.leaky_relu,
                 ):
        super(Attention, self).__init__()
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.activation_fnx = activation_fnx
        self.fc_reduceheads = nn.Linear(embedding_dim, embedding_dim)
        # self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, query, context, memory):
        if query.size(1) != context.size(1):
            # query = np.tile(query, (context.size(1),1))
            query = query.repeat(1, context.size(1), 1)
        qc_logits = self.fc_logit(torch.tanh(context + query))
        attention_probs = F.softmax(qc_logits, dim=1)
        attention_heads = (memory * attention_probs).sum(1).unsqueeze(1)
        attention_heads = self.activation_fnx(attention_heads)
        attention_result = self.fc_reduceheads(attention_heads)

        return attention_result


class AttentiveGraphPooling(nn.Module):
    def __init__(self,
                 embedding_dim,
                 init_w=3e-3
                 ):
        super(AttentiveGraphPooling, self).__init__()
        # self.input_independent_query = torch.tensor(embedding_dim, requires_grad=True, dtype=torch.float32, device='cpu')
        # self.input_independent_query.data.uniform_(-init_w, init_w)
        self.attention = Attention(embedding_dim)

    def forward(self, response_embeddings):
        embedding_dim = 64
        init_w = 3e-3
        input_independent_query = Parameter(torch.Tensor(embedding_dim))
        input_independent_query.data.uniform_(-init_w, init_w)

        N = response_embeddings.size(0)
        query = input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)
        context = response_embeddings
        memory = response_embeddings
        attention_result = self.attention(query, context, memory)

        return attention_result

class Mlp(nn.Module):
    def __init__(self,
                 env_params,
                 embedding_dim,
                 init_w=3e-3
                 ):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.last_fc = nn.Linear(256, env_params['action'])
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        # self.layer_norm1 = nn.LayerNorm(64)
        # self.layer_norm2 = nn.LayerNorm(64)
        # self.layer_norm3 = nn.LayerNorm(64)
        # self.last_fc_log_std = nn.Linear(64, env_params['action'])
        # self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        # self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs_input):
        h = obs_input
        h = F.relu(self.fc1(h))
        # h = self.layer_norm1(h)
        h = F.relu(self.fc2(h))
        # h = self.layer_norm2(h)
        h = F.relu(self.fc3(h))
        # h = self.layer_norm3(h)

        action = torch.tanh(self.last_fc(h))

        # mean = self.last_fc(h)
        # log_std = self.last_fc_log_std(h)
        # std = torch.exp(log_std)
        # z = (
        #         mean +
        #         std *
        #         Normal(
        #             torch.zeros(mean.size()),
        #             torch.ones(std.size())
        #         ).sample()
        # )
        # z.requires_grad_()
        # action = torch.tanh(z)

        return action









