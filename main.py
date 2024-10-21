import argparse
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from smac.env import StarCraft2Env

'transformers.DistilBertTokenizer',
'transformers.DistilBertForSequenceClassification',
'transformers.DistilBertConfig',

import numpy as np
import threading


class Buffer:
    def __init__(self, args, obs_token_size):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.obs_shape = obs_token_size
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

# 动作选择
def choose_action(decesion, hidden_state, avail_actions, epsilon, evaluate=False):
    avail_actions_ind = np.nonzero(avail_actions)[0] 
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
    q_value = decesion(hidden_state)
    q_value[avail_actions == 0.0] = - float("inf")
    if np.random.uniform() < epsilon:
        action = np.random.choice(avail_actions_ind) 
    else:
        action = torch.argmax(q_value).detach().cpu()
    return action

# 决策层：三层MLP构成
class Decesion(nn.Module):
    def __init__(self, config, args):
        super(Decesion, self).__init__()
        input_shape = config.dim
        self.args = args
        self.pre_actor = nn.Linear(input_shape, args.hidden_dim)
        self.hidden = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.actor = nn.Linear(args.hidden_dim, args.n_actions)
        self.dropout = nn.Dropout(0.2)

    def forward(self, obs):
        x = f.relu(self.pre_actor(obs))
        pooled_output = self.hidden(x) 
        pooled_output = nn.ReLU()(pooled_output) 
        pooled_output = self.dropout(pooled_output)  
        q = self.actor(pooled_output)
        return q

# 动作2nlp字典
avail_actions_dict = {
     0: "no operation",
     1: "stop",
     2: "north",
     3: "south",
     4: "east",
     5: "west"
}

# 动作转换nlp
def translate_avail_actions(avail_actions):
    str = ""
    for i in range(6):
        if avail_actions[i] == 1:
            str += avail_actions_dict[i] + ","
    for j in range(len(avail_actions[6:])):
        if avail_actions[j] == 1:
             str += f"enemy{j},"
    return str

# 转换obs2text
def translate_obs(env, agent_id):
    language_obs = ""
    unit = env.get_unit_by_id(agent_id)
    if unit.health > 0:
        language_obs += f"[CLS]The observation of agent{agent_id}:[SEP]"
        x, y = unit.pos.x, unit.pos.y
        own_position = f"position is {x}, {y}"
        available_actions = env.get_avail_agent_actions(agent_id)
        translate_available_actions = translate_avail_actions(available_actions)
        translate_own_info = f"own informations are:[{own_position}], available actions are [{translate_available_actions}], type is soldier"
        if env.obs_own_health:
            health = unit.health / unit.health_max
            translate_own_info += f", health is {health}"
            if env.shield_bits_ally > 0:
                max_shield = env.unit_max_shield(unit)
                shield = unit.shield / max_shield
                translate_own_info += f", shield is {shield}"
        translate_own_info += ".[SEP]"
        language_obs += translate_own_info
        for e_id, e_unit in env.enemies.items():
            e_x = e_unit.pos.x
            e_y = e_unit.pos.y
            sight_range = env.unit_sight_range(agent_id)
            dist = env.distance(unit.pos.x, unit.pos.y, e_x, e_y)
            translate_enemy_info = ""
            if dist < sight_range and e_unit.health > 0:
                avail_attack = "attackable" if available_actions[env.n_actions_no_attack + e_id] == 1 else "not attackable"
                translate_enemy_info += f"(enemy{e_id}{avail_attack}, position is {e_x}, {e_y}, {dist}, type is soldier"
                if env.obs_all_health:
                    health = e_unit.health / e_unit.health_max
                    translate_enemy_info += f", health is {health}"
                    if env.shield_bits_enemy > 0:
                        max_shield = env.unit_max_shield(e_unit)
                        shield = e_unit.shield / max_shield
                        translate_enemy_info += f", shield is {shield}"
                translate_enemy_info += ")"
        language_obs += f"enemy informations are:{translate_enemy_info}.[SEP]"

        al_ids = [
                al_id for al_id in range(env.n_agents) if al_id != agent_id
            ]
        for i, al_id in enumerate(al_ids):
            al_unit = env.get_unit_by_id(al_id)
            al_x = al_unit.pos.x
            al_y = al_unit.pos.y
            dist = env.distance(x, y, al_x, al_y)
            translate_partner_info = ""
            if dist < sight_range and al_unit.health > 0:
                translate_partner_info += f"(partner{i}, position is {al_x}, {al_y}, {dist}, type is soldier" 
                if env.obs_all_health:
                    health = al_unit.health / al_unit.health_max
                    translate_partner_info += f", health is {health}"
                    if env.shield_bits_ally > 0:
                        max_shield = env.unit_max_shield(al_unit)
                        shield = al_unit.shield / max_shield
                        translate_partner_info += f", shield is {shield}"
                    translate_partner_info += ")"
        language_obs += f"partner informations are:{translate_enemy_info}.[SEP]" 
    return language_obs

def train_decesion(batch, decesion, target_decesion, scheduler, train_epoch):
    def _get_max_episode_len(batch, args):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = args.episode_limit
        return max_episode_len

    episode_num = batch['o'].shape[0]
    for key in batch.keys():  # 把batch里的数据转化成tensor
        if key == 'u':
            batch[key] = torch.tensor(batch[key], dtype=torch.long).cuda()
        else:
            batch[key] = torch.tensor(batch[key], dtype=torch.float32).cuda()
    o, o_next, u, r, avail_u, avail_u_next, terminated = batch['o'], batch['o_next'], batch['u'], \
                                                            batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                            batch['terminated']
    mask = 1 - batch["padded"].float() 

    q_evals = decesion(o)
    q_evals[avail_u == 0.0] = - float("inf")
    q_targets = target_decesion(o_next)
    q_evals[avail_u_next == 0.0] = - float("inf")
    q_evals = torch.gather(q_evals, dim=-1, index=u).squeeze(3)
    q_targets = q_targets.max(dim=-1)[0]


    targets = r + 0.99 * q_targets * (1 - terminated)
    td_error = (q_evals - targets.detach())
    masked_td_error = mask * td_error  
    loss = (masked_td_error ** 2).sum() / mask.sum()
    scheduler.step(loss)
    if train_epoch > 0 and train_epoch % train_interval == 0:
        target_decesion.load_state_dict(decesion.state_dict())


def evaluate(decesion, env, args):
    rewards = 0
    win_tags = 0
    evaluate_times = 8
    for i in range(evaluate_times):
        terminated = False
        win_tag = False
        episode_reward = 0
        episode_step = 0
        env.reset()
        # 每一个episode推演
        while not terminated and episode_step < args.episode_limit:
            actions = []
            for agent_id in range(env.n_agents):
                avail_action = env.get_avail_agent_actions(agent_id)
                obs = translate_obs(env, agent_id)
                obs_token = tokenizer(obs)
                input_ids = torch.tensor(obs_token["input_ids"]).reshape(1, -1).cuda()
                attention_mask = torch.tensor(obs_token["attention_mask"]).reshape(1, -1).cuda()
                output = LM(input_ids=input_ids, attention_mask=attention_mask)
                hidden_state = output[0]
                hidden_state = hidden_state[:, 0]
                action = choose_action(decesion, hidden_state, avail_action, 0) 
                actions.append(np.int32(action))

            reward, terminated, info = env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False   
            episode_reward += reward
            episode_step += 1

        if win_tag:
                win_tags += 1
        rewards += episode_reward
    win_tags /= evaluate_times
    rewards /= evaluate_times
    print("*evaluate results: Win {:}, Rewards {:.2f}.".format(win_tags, rewards))

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
parser.add_argument('--map', type=str, default='3m', help='the map of the game')
parser.add_argument('--seed', type=int, default=7, help='random seed')
parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action 8')
parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
parser.add_argument('--hidden_dim', type=int, default=128, help='how many steps to make an action 8')
parser.add_argument('--buffer_size', type=int, default=5000, help='how many steps to make an action 8')
parser.add_argument('--lr', type=float, default=1e-4, help='how many steps to make an action 8')
args = parser.parse_args()

env = StarCraft2Env(
                    map_name=args.map,
                    step_mul=args.step_mul,
                    difficulty=args.difficulty,
                    game_version=args.game_version,
                    replay_dir=args.replay_dir
                    )
env_info = env.get_env_info()
args.n_actions = env_info["n_actions"]
args.n_agents = env_info["n_agents"]
args.state_shape = env_info["state_shape"]
args.obs_shape = env_info["obs_shape"]
args.episode_limit = env_info["episode_limit"]

# 语言模型和决策模型
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel
pretrained_path = "./distilbert_pretrained"
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_path, return_tensors='pt')
LMConfig = DistilBertConfig.from_pretrained(pretrained_path)
LM = DistilBertModel.from_pretrained(pretrained_path, config=LMConfig).cuda()
# model = LM.to(args.device)
decesion = Decesion(LMConfig, args).cuda()
target_decesion = Decesion(LMConfig, args).cuda()
buffer = Buffer(args, LMConfig.dim)
optimizer = torch.optim.Adam(decesion.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

# 推演入口
total_steps = 100000
steps = 0
epsilon = 0.9
min_epsilon = 0.05
train_epochs = 7
batch_size = 16
train_epoch = 0
train_interval = 300
evaluate_interval = 5000
evaluate_epoch = 0

while steps < total_steps:
    terminated = False
    win_tag = False
    episode_step = 0
    episode_reward = 0
    o, u, r, avail_u, terminate, padded = [], [], [], [], [], []
    env.reset()
    # 每一个episode推演
    while not terminated and episode_step < args.episode_limit:
        actions, avail_actions, obs_tokens = [], [], []
        for agent_id in range(env.n_agents):
            avail_action = env.get_avail_agent_actions(agent_id)
            obs = translate_obs(env, agent_id)
            obs_token = tokenizer(obs)
            input_ids = torch.tensor(obs_token["input_ids"]).reshape(1, -1).cuda()
            attention_mask = torch.tensor(obs_token["attention_mask"]).reshape(1, -1).cuda()
            output = LM(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output[0]
            hidden_state = hidden_state[:, 0]
            action = choose_action(decesion, hidden_state, avail_action, epsilon) 
            actions.append(np.int32(action))
            avail_actions.append(avail_action)
            obs_tokens.append(hidden_state)

        obs_tokens = torch.cat(obs_tokens).detach().cpu().numpy()
        # collect trajactory
        reward, terminated, info = env.step(actions)
        o.append(obs_tokens)
        u.append(np.reshape(actions, [args.n_agents, 1]))
        avail_u.append(avail_actions)
        r.append([reward])
        terminate.append([int(terminated)])
        padded.append([0.])
        win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
        episode_reward += reward
        episode_step += 1
        epsilon = epsilon - 0.00015 if epsilon > min_epsilon else epsilon  
    
    actions, avail_actions, obs_tokens = [], [], []
    for agent_id in range(env.n_agents):
        avail_action = env.get_avail_agent_actions(agent_id)
        obs = translate_obs(env, agent_id)
        obs_token = tokenizer(obs)
        input_ids = torch.tensor(obs_token["input_ids"]).reshape(1, -1).cuda()
        attention_mask = torch.tensor(obs_token["attention_mask"]).reshape(1, -1).cuda()
        output = LM(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        hidden_state = hidden_state[:, 0]
        avail_actions.append(avail_action)
        obs_tokens.append(hidden_state)

    obs_tokens = torch.cat(obs_tokens).detach().cpu().numpy()
    o.append(obs_tokens)
    avail_u.append(avail_actions)
    o_next = o[1:]
    o = o[:-1]
    avail_u_next = avail_u[1:]
    avail_u = avail_u[:-1]

    # if step < self.episode_limit，padding
    for i in range(episode_step, args.episode_limit):
        o.append(np.zeros((args.n_agents, LMConfig.dim)))
        u.append(np.zeros([args.n_agents, 1]))
        r.append([0.])
        o_next.append(np.zeros((args.n_agents, LMConfig.dim)))
        avail_u.append(np.zeros((args.n_agents, args.n_actions)))
        avail_u_next.append(np.zeros((args.n_agents, args.n_actions)))
        padded.append([1.])
        terminate.append([1.])

    episode = dict( 
                    o=np.expand_dims(np.array(o), axis=0),
                    u=np.expand_dims(np.array(u),axis=0),
                    r=np.expand_dims(np.array(r),axis=0),
                    avail_u=np.expand_dims(np.array(avail_u),axis=0),
                    o_next=np.expand_dims(np.array(o_next),axis=0),
                    avail_u_next=np.expand_dims(np.array(avail_u_next),axis=0),
                    padded=np.expand_dims(np.array(padded),axis=0),
                    terminated=np.expand_dims(np.array(terminate),axis=0),
                    )
    buffer.store_episode(episode)

    for _ in range(train_epochs):
        train_epoch += 1
        mini_batch = buffer.sample(min(buffer.current_size, batch_size))
        train_decesion(mini_batch, decesion, target_decesion, scheduler, train_epoch)

    steps += episode_step
    if (steps // evaluate_interval) > evaluate_epoch:
        evaluate(decesion, env, args)
        evaluate_epoch += 1
    print("total step:", steps) 