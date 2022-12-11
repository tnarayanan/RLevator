import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

experiment = "3floor"

df = None
df_curr = None

if experiment == "3floor":
    df0 = pd.read_csv("results_data/env_v7_elev1-1_floor3-3_rand0_16e031_PPO_1.csv")
    df1 = pd.read_csv("results_data/env_v7_elev1-1_floor3-3_rand1_9c5894_PPO_1.csv")
    df2 = pd.read_csv("results_data/env_v7_elev1-1_floor3-3_rand2_fadd18_PPO_1.csv")
    df = pd.concat((df0, df1, df2))
elif experiment == "5floor":
    df0 = pd.read_csv("results_data/env_v7_elev1-1_floor5-5_rand0_2e083c_PPO_1.csv")
    df1 = pd.read_csv("results_data/env_v7_elev1-1_floor5-5_rand1_d8696d_PPO_1.csv")
    df2 = pd.read_csv("results_data/env_v7_elev1-1_floor5-5_rand2_0cacd7_PPO_1.csv")
    df = pd.concat((df0, df1, df2))
elif experiment == "8floor":
    df0 = pd.read_csv("results_data/env_v7_elev1-1_floor8-8_rand0_c068ad_PPO_1.csv")
    df1 = pd.read_csv("results_data/env_v7_elev1-1_floor8-8_rand1_65a3a2_PPO_1.csv")
    df2 = pd.read_csv("results_data/env_v7_elev1-1_floor8-8_rand2_490fbe_PPO_1.csv")

    df0_curr = pd.read_csv("results_data/env_v7_elev1-1_floor3-8_rand0_0e7c72_PPO_1.csv")
    df1_curr = pd.read_csv("results_data/env_v7_elev1-1_floor3-8_rand1_09feb9_PPO_1.csv")
    df2_curr = pd.read_csv("results_data/env_v7_elev1-1_floor3-8_rand2_2a5d7c_PPO_1.csv")

    df = pd.concat((df0, df1, df2))
    df['Training method'] = 'Without curriculum learning'
    df_curr = pd.concat((df0_curr, df1_curr, df2_curr))
    df_curr['Training method'] = 'With curriculum learning'
    df = pd.concat((df, df_curr))

df['Step'] /= 100

if df_curr is not None:
    sns.lineplot(data=df, x='Step', y='Value', hue='Training method')
else:
    sns.lineplot(data=df, x='Step', y='Value')

plt.title(f"RLevator episodic reward during training: {experiment[0]} floors")
plt.xlabel("Training Episode")
plt.ylabel("Episodic Reward")
# plt.show()
plt.savefig(f"results_figures/{experiment}_reward.pdf")