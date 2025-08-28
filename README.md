
# 📘Tutorial
1. [Model-Based Methods Tutorial (ICML 2020)](https://sites.google.com/view/mbrl-tutorial)  
2. [Model-Based RL Blog (Georgiev, 2023)](https://www.imgeorgiev.com/2023-11-16-mbrl/)  
3. [CS 4789/5789 – Lec10 by Sarah Dean](https://vod.video.cornell.edu/media/CS+4789A+Lecture+10/1_ymki7oc8)  
4. [Practical Model-Based Algorithms for Reinforcement Learning and Imitation Learning, with Theoretical Analyses – Tengyu Ma (Simons Institute, July 15, 2019)](https://simons.berkeley.edu/talks/practical-model-based-algorithms-reinforcement-learning-imitation-learning-theoretical)    
5. [Understanding and Improving Model-Based Deep Reinforcement Learning, Jessica Hamrick](https://www.youtube.com/watch?v=o9ji8jAcSx4)
6. [The challenges of model-based reinforcement learning and how to overcome them - Csaba Szepesvári](https://www.youtube.com/watch?v=-Y-fHsPIQ_Q)
7. [Model-Based Reinforcement Learning: Theory and Practice Michael Janner](https://bair.berkeley.edu/blog/2019/12/12/mbpo/)
8. [Awesome MBRL](https://github.com/opendilab/awesome-model-based-RL)
9. [Bayesian Reinforcement Learning](https://bayesianrl.github.io/)


# 📘Model error 
1. Investigating Compounding Prediction Errors in Learned Dynamics Models    
2. PILCO: A Model-Based and Data-Effcient Approach to Policy Search, ICML 2011 (Model bias: accumulation of singel step model error systematically over time.)    
3. Plan To Predict: Learning an Uncertainty-Foreseeing Model for Model-Based Reinforcement Learning, NIPS 2022 (single-step vs multi-step prediction loss)
4. 


# 📘Uncertainty in MBRL
1. Self-Supervised Exploration via Disagreement, ICML 2019    
2. Sample Efficient Reinforcement Learning via Model-Ensemble Exploration and Exploitation    



# 📘Return Bound Design
1. Algorithmic Framework for Model-Based Deep Reinforcement Learning with Theoretical Guarantees(SLBO) – Meta AI, ICLR 2019

# 📘Predictive Uncertainty Estimation
1. Aleatoric and Epistemic Uncertainty in Machine Learning - https://www.gdsd.statistik.uni-muenchen.de/2021/gdsd_huellermeier.pdf
2. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods, Machine Learning 2021, Springer Nature    
3. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles - NIPS 2017 (scalable replacemnent for Bayesain NNs, Spread across ensemble predictions → epistemic| Each network’s predicted variance → aleatoric.)
                  <img width="728" height="289" alt="image" src="https://github.com/user-attachments/assets/5eb01438-b19e-4886-bcb8-e39c407a21cc" />
  
4. T. G. Dietterich. Ensemble methods in machine learning. In Multiple classifier systems. 2000 (Shows ensembles (model combination) improve model prdictve performance)
5. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, ICML 2016 (MC dropout: Approximates Bayesian inference for compute cheap predicitve uncertainty estimate. Handles only epsitemic uncertainty)
6. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? , NIPS 2017 (extended the MC dropout approach to also learn aleatoric uncertainty via a variance output head, making it possible to quantify both epistemic + aleatoric in a single model.)    
7. Yarin Gal. Uncertainty in deep learning. PhD Thesis, PhD thesis, University of Cambridge, 2016.
8. Armen Der Kiureghian and Ove Ditlevsen. Aleatory or epistemic? Does it matter? Structural Safety,  31(2):105–112, 2009.    
  

# 📘Predictive Propgation during model rollout

# 📘Neat history - check related works in 
1. Modified PETS
2. Trust the Model Where It Trusts Itself- Model-Based Actor-Critic with Uncertainty-Aware Rollout Adaption
3. On Rollouts in Model-Based Reinforcement Learning




# 📘Model Calibration
1. A. P. Dawid. The well-calibrated Bayesian. Journal of the American Statistical Association, 1982
2. M. H. DeGroot and S. E. Fienberg. The comparison and evaluation of forecasters. The statistician, 1983.
3. Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning, NIPS 2020        
4. Accurate Uncertainties for Deep Learning Using Calibrated Regression, ICML 2018
5.  Near-optimal Regret Bounds for Reinforcement Learning,  NIPS 2009 (Error radius is learned from data (concentration inequalities → Hoeffding, Bernstein, GP posteriors))      


# 📘Exploration
1. [Exploration and Exploitation-10703 Deep Reinforcement Learning	and	Control](https://www.cs.cmu.edu/~rsalakhu/10703/Lectures/Lecture_Exploration.pdf)    
## Naïve random exploration (best suited for tabular data, compute costly for large dimensional problem)
1. ε-greedy → Watkins (1989), Sutton & Barto (1998/2018).
2. Softmax/Boltzmann → Sutton & Barto, Kaelbling et al. (1996), Thrun (1992).
## Directed exploration- Uncertainty-driven exploration focuses exploration where where the agent’s knowledge is ambiguous.
1. Optimism in the Face of Uncertainty (OFU)    
2. Posterior (Thompson) Sampling
3. Intrinsic Motivation / Bonus-Based
## Optimistic exploration papers
1. [The need for Explicit Exploration in Model-based Reinforcement Learning](https://berkenkamp.me/blog/2020-12-06-mbrl-exploration/#Mania2019Certainty)
2. H-UCRL - Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning, NIPS 2020
3. UCRL - Near-optimal Regret Bounds for Reinforcement Learning, JMLR 2010 (Tabular MDPS)
4. R-Max - A General Polynomial Time Algorithm for Near-Optimal Reinforcement Learning, JMLR 2023 (Tabular MDPS)
5. GP-UCRL - Online Learning in Kernelized Markov Decision Processes (continuous space)      
6. Regret Bounds for the Adaptive Control of Linear Quadratic Systems, JMLR 2011    
7. Trust-region UCRL meta-algorithm (SLBO) - Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees, ICLR 2019
8. Optimism-driven exploration for nonlinear systems - ICRA, 2015 (OFU based work extended for deep MBRL)
9. Safe Exploration in Reinforcement Learning: Theory and Applications in Robotics
10. The Many Faces of Optimism: a Unifying Approach, ICML 2008    
11. Areas to explore: Computational tractability of strict OFU, combining OFU with safety guarantees
## Optimistic/Pessimistic safe exploration papers
1. DOPE: Doubly Optimistic and Pessimistic Exploration for Safe Reinforcement Learning
2. A Policy Gradient Primal-Dual Algorithm for Constrained MDPs with Uniform PAC Guarantees
3. Safe Reinforcement Learning for Constrained Markov Decision Processes with Stochastic Stopping Time, CDC 2024
4. A safe exploration approach to constrained Markov decision processes
5. Ensuring Safety in an Uncertain Environment: Constrained MDPs via Stochastic Thresholds
6. A Constructive Review of Safe Exploration in Reinforcement Learning
7. Constrained Reinforcement Learning Under Model Mismatch
8. OptCMDP: Exploration-Exploitation in Constrained MDPs
9. Optless: Learning Policies with Zero or Bounded Constraint Violation for Constrained MDPs, ICML 2021   



# 📘Benchmark Analysis
1. Benchmarking Deep Reinforcement Learning for Continuous Control, ICML 2016    
2. Reinforcement Learning with Deep Energy-Based Policies, ICML 2017    



# 📘Toolbox
1. [MBRL-Lib Meta AI](https://github.com/facebookresearch/mbrl-lib). [Paper : MBRL-Lib: A Modular Library forModel-based Reinforcement Learning](https://arxiv.org/pdf/2104.10159)
2. [DI-engine](https://github.com/opendilab/DI-engine)
3. [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)






