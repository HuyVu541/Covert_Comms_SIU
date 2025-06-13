import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc
import math
import pickle
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Use plain tqdm for .py files

def path_loss(d, f=2.4): # d is in kilometre
    return 20 * np.log10(d) + 20 * np.log10(f) + 92.45

def convert_dBm_W(P):
    return 10**((P-30) / 10)

def convert_dB_W(P):
    return 10**((P) / 10)

def watts_to_dB(P):
    return 10 * np.log10(P)

noise = 1e-13
h_uu = 0.5
rho = 1e-10
tau = 1
relay_num = 5 #5, 10, 20, 30
uav_h = 2

from adjustText import adjust_text  # Import adjustText library

def generate_map(relay_num, random_seed=600):
    # Fixed points
    source = np.array([[0, 0]])
    destination = np.array([[2, 2]])

    # Parameters
    map_size = 2  # Map boundary

    # Generate unique random seed for each relay based on its index
    np.random.seed(random_seed)  # Set base seed for reproducibility
    relay_points = []

    uav_point = np.random.rand(1, 2) * map_size
    
    for i in range(relay_num):
        # Set each relay's seed based on its index
        np.random.seed(random_seed + 1 + i)
        relay_points.append([(np.random.rand(1) * map_size)[0], (np.random.rand(1) * map_size)[0]])
    
    relay_points = np.vstack(relay_points)

    # Combine all points
    all_points = np.vstack([source, relay_points, uav_point, destination])
    labels = ['Source'] + [f'Relay{i+1}' for i in range(relay_num)] + ['UAV', 'Destination']

    # # Plot
    # plt.figure(figsize=(8, 8))
    # plt.scatter(source[:, 0], source[:, 1], c='green', label='Source', marker='s', s=30)
    # plt.scatter(destination[:, 0], destination[:, 1], c='red', label='Destination', marker='s', s=30)
    # plt.scatter(relay_points[:, 0], relay_points[:, 1], c='blue', label='Relays', marker='o')
    # plt.scatter(uav_point[:, 0], uav_point[:, 1], c='orange', label='UAV', marker='^', s=30)

    # # Annotate with dynamic label adjustment
    # texts = []
    # for i, (x, y) in enumerate(all_points):
    #     text = plt.text(x + 0.05, y + 0.05, labels[i], fontsize=9)
    #     texts.append(text)
    
    # Use adjustText to adjust the positions of the labels
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

    # plt.xlim(-0.25, map_size + 0.25)
    # plt.ylim(-0.25, map_size + 0.25)
    # plt.grid(True)

    # # Place the legend in the top-right corner
    # plt.legend(loc='upper left', fontsize=10)

    # plt.title('Map of Source, Relays, UAV, and Destination')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.savefig('HD Map.png', dpi=600, bbox_inches='tight')

    # plt.show()

    dist_matrix = cdist(all_points, all_points)

    idx_source = 0
    idx_relays_start = 1
    idx_relays_end = idx_relays_start + relay_num
    idx_uav = idx_relays_end
    idx_destination = idx_uav + 1

    # Extract required distances
    d_s_ri = dist_matrix[idx_source, idx_relays_start:idx_relays_end]      # Source to each relay
    d_u_ri = (np.array(dist_matrix[idx_uav, idx_relays_start:idx_relays_end])**2 + uav_h**2)**0.5         # UAV to each relay
    d_ri_u = d_u_ri
    d_ri_d = dist_matrix[idx_relays_start:idx_relays_end, idx_destination] # Each relay to Destination
    d_u_s = (dist_matrix[idx_uav, idx_source]**2 + uav_h**2)**0.5                               # UAV to Source
    d_u_d = (dist_matrix[idx_uav, idx_destination]**2 + uav_h**2)**0.5    

    n = relay_num + 1
    
    pl_s_u = path_loss(d_u_s)                          # Path loss from source to UAV
    pl_ri_u = np.array([path_loss(d) for d in d_ri_u])
    pl_u_ri = pl_ri_u
    pl_s_ri = np.array([path_loss(d) for d in d_s_ri])
    pl_ri_d = np.array([path_loss(d) for d in d_ri_d])  # Path loss from relay to destination
    pl_u_d = path_loss(d_u_d)            

    return d_s_ri, d_u_ri, d_ri_u, d_ri_d, d_u_s, d_u_d, n, pl_s_u, pl_ri_u, pl_u_ri, pl_s_ri, pl_ri_d, pl_u_d

d_s_ri, d_u_ri, d_ri_u, d_ri_d, d_u_s, d_u_d, n, pl_s_u, pl_ri_u, pl_u_ri, pl_s_ri, pl_ri_d, pl_u_d = generate_map(relay_num=relay_num)

pl_s_u = convert_dB_W(pl_s_u)
pl_ri_u = convert_dB_W(pl_ri_u)
pl_u_ri = convert_dB_W(pl_u_ri)
pl_s_ri = convert_dB_W(pl_s_ri)
pl_ri_d = convert_dB_W(pl_ri_d)
pl_u_d = convert_dB_W(pl_u_d)

P_u_start = 25
P_u_end = 45
P_x_start = 25
P_x_end = 45

def math_detection_prob(P_x, P_u, chosen_r, m_relay = 2, m_source = 2, tau = 1, loops = 100):
    P_s = P_x 
    P_ri = P_x 

    theta_s_u = 1 / m_source     
    theta_ri_u = 1 / m_relay   

    beta_s_u = theta_s_u * P_s / pl_s_u      # Source to UAV
    beta_ri_u = (theta_ri_u * P_ri / pl_ri_u)[chosen_r]  # Relay to UAV

    beta_min = min([beta_s_u, beta_ri_u])

    coef = np.prod([(beta_min / beta_ri_u)**m_relay, (beta_min / beta_s_u)**m_source])

    sigmas = [1]
    for k in range(1, loops):
        epsilons = []
        for j in range(1, k+1):
            epsilon = 0
            epsilon += (m_relay/j) * ((1 - beta_min / beta_ri_u)**j)
            epsilon += (m_source/j) * ((1 - beta_min / beta_s_u)**j)
            epsilons.append(epsilon)
        sigma = (1/k) * sum([j * epsilons[j-1] * sigmas[k-j] for j in range(1, k+1)])
        sigmas.append(sigma)
    res = []
    lambda_ = rho * P_u * h_uu

    tau = tau * lambda_
    P_detection_cdf = 1 - (coef * sum([sigmas[k] * gammainc((m_source + m_relay) + k, tau/beta_min) for k in range(len(sigmas))]))
    return P_detection_cdf

m_relay = 2
m_source = 2
num_samples = 10**7
h_ri_u = np.array(np.random.gamma(m_relay, 1/ m_relay, num_samples))
h_s_u = np.array(np.random.gamma(m_source, 1/ m_source, num_samples))

def sim_detection_prob(P_x, P_u, chosen_r, m_relay = 2, m_source = 2, num_samples = 10**7):
    P_s = P_x #dBm
    P_ri = P_x  # Power at relay
    
    numerator = 0

    numerator += P_s * h_s_u / pl_s_u
    numerator += P_ri * h_ri_u / pl_ri_u[chosen_r]

    temp = numerator / (noise + rho * P_u * h_uu) 
    return np.mean(temp > tau)

# Phase 1 Outage Probability
def phase1_SINR(P_s, P_u, m_relay = 2, m_source = 2, num_samples = 10000):
    P_s = convert_dBm_W(P_s)
    P_u = convert_dBm_W(P_u)
    
    theta_s_ri = np.array([P_s / pl for pl in pl_s_ri])    # Source to relay
    theta_u_ri = np.array([P_u / pl for pl in pl_u_ri])    # UAV to relay

    h_s_ri = np.array([np.random.gamma(m_relay, 1/m_relay, num_samples) for i in range(relay_num)])
    h_u_ri = np.array([np.random.gamma(m_relay, 1/m_relay, num_samples) for i in range(relay_num)])
    
    SINRs = []
    for ri in range(relay_num):
        SINR = (P_s * h_s_ri[ri] / pl_s_ri[ri]) / (noise + P_u * h_u_ri[ri] / pl_u_ri[ri]) 
        SINRs.append(SINR)
    SINRs = np.array(SINRs)
    # return SINRs
    return SINRs

# Phase 2 Outage Probability
def phase2_SINR(P_ri, P_u, m_relay = 2, m_source = 2, num_samples = 10000):
    P_ri = convert_dBm_W(P_ri)
    P_u = convert_dBm_W(P_u)
 
    theta_ri_d = np.array([P_ri / pl for pl in pl_ri_d])   # Source to relay
    theta_u_d = P_u / pl_u_d     # UAV to relay

    h_ri_d = np.array([np.random.gamma(m_relay, 1/m_relay, num_samples) for i in range(relay_num)])
    h_u_d = np.random.gamma(m_relay, 1/m_relay, num_samples)

    SINRs = []
    for ri in range(relay_num):
        SINR = (P_ri * h_ri_d[ri] / pl_ri_d[ri]) / (noise + P_u * h_u_d / pl_u_d) 
        # SINR = (P_ri * h_ri_d[ri] / pl_ri_d[ri]) / (noise + P_u * h_u_d / pl_u_d) 
        SINRs.append(SINR)
    SINRs = np.array(SINRs)

    return SINRs

def choose_relay_2(P_x, P_u):
    SINRs = list(zip(phase2_SINR(30,30,num_samples=1).reshape(-1), phase1_SINR(30,30,num_samples=1).reshape(-1)))
    min_SINRs = [min(i) for i in SINRs]
    return min_SINRs.index(max(min_SINRs))

import numpy as np
import pickle
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Worker function
def process_pair(P_x, P_u):
    Px_w = convert_dBm_W(P_x)
    Pu_w = convert_dBm_W(P_u)
    temp_r = [choose_relay_2(P_x, P_u) for _ in range(100)]
    chosen_relays = pd.Series(temp_r).value_counts().items()

    total_math = 0
    total_sim = 0
    for r, count in chosen_relays:
        total_math += math_detection_prob(Px_w, Pu_w, r) * count
        total_sim += sim_detection_prob(Px_w, Pu_w, r) * count

    return total_math / 100, total_sim / 100

def unpack_and_process(args):
    return process_pair(*args)

if __name__ == "__main__":
    # Grid setup
    P_x_vals = np.arange(P_x_start, P_x_end + 0.1, 0.1)
    P_u_vals = np.arange(P_u_start, P_u_end + 0.1, 0.1)

    num_x = len(P_x_vals)
    num_u = len(P_u_vals)

    P_detection_math = np.zeros((num_x, num_u))
    P_detection_sim = np.zeros((num_x, num_u))
    
    tasks = [(P_x, P_u) for P_x in P_x_vals for P_u in P_u_vals]
    results = [None] * len(tasks)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pair, *task): idx for idx, task in enumerate(tasks)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            results[idx] = future.result()

    # Fill 2D arrays from flat result list
    for idx, (math_val, sim_val) in enumerate(results):
        i = idx // num_u
        j = idx % num_u
        P_detection_math[i, j] = math_val
        P_detection_sim[i, j] = sim_val

    # Save
    with open(f'checkpoint_det_{relay_num}_relay_(math)_T.pkl', 'wb') as f:
        pickle.dump(P_detection_math, f)
    with open(f'checkpoint_det_{relay_num}_relay_(sim)_T.pkl', 'wb') as f:
        pickle.dump(P_detection_sim, f)
