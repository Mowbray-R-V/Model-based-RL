
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
10. [TalK RL - Nathan Lambert on Model-based RL](https://www.talkrl.com/episodes/nathan-lambert)    
11. 

# 📘Types
## Polciy learning using synthetic data (stored in buffer)
1. Dyna, an integrated architecture for learning, planning, and reacting, ACM 1991 (**One-step rollout**)
2. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees, ICLR 2019 (**Multi-step rollout**)
4. When to trust your model: Model-based policy optimization, ICML  2019 (**Multi-step rollout**)
## Value estimation using longer rollouts rather than 1-step target (not stored in buffer)    
1. Model-based value estimation for efficient model-free reinforcement learning, 2018
2. Sample-efficient reinforcement learning with stochastic ensemble value expansion, NIPS 2018

# 📘Adaptive rollout
1. Dynamic-Horizon Model-Based Value Estimation With Latent Imagination, IEEE TNLS 2024 (**MVE style algorithm with adaptive rollout in world model**)   

# 📘Latent space models for POMDPS 
Latent state variables serve as a belief state ≈ agent’s best guess of the hidden true state. EX: Visual tasks (Atari, DMControl from pixels): the raw image doesn’t tell you velocity or hidden forces. Latent state encodes those quantities.
1. [POMDPs for Dummies](https://www.pomdp.org/tutorial/pomdp-solving.html)    
2. World Models,  2018
3. Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Model, NIPs 2020 (**SAC for POMDPs**)
4. Value Prediction Network, NIPS 2017
5. Mastering Atari with Discrete World Models, ICLR 2021 (**DreamerV2**)
6. Dream to Control: Learning Behaviors by Latent Imagination, ICML 2020 (**Dreamer**)
7. Learning Latent Dynamics for Planning from Pixels, ICML 2019 (**Planet**)
8. Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images,  2015
9. Temporal Difference Learning for Model Predictive Control, ICML 2022  (**TD-MPC**)
10. TD-MPC2:Scalable, Robust World Models for Continuous Control, ICLR 2024          



# 📘Model error 
1. Investigating Compounding Prediction Errors in Learned Dynamics Models    
2. PILCO: A Model-Based and Data-Effcient Approach to Policy Search, ICML 2011 (Model bias: accumulation of singel step model error systematically over time.)    
3. Plan To Predict: Learning an Uncertainty-Foreseeing Model for Model-Based Reinforcement Learning, NIPS 2022 (single-step vs multi-step prediction loss)
4. 

# 📘Monotoic policy update guarantess
1. Approximately optimal approximate reinforcement learning, ICML 2002 (proposes a lower-bound optimization framework for mixture policies to guarantees monotonic improvement)
2. Trust Region Policy Optimization, ICML 2015 (extends this for stochastic policies by introducing a trust region constraint based on a divergence measure between the old and updated policies for stable updates)
3. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees, ICLR 2019
4. When to trust your model: Model-based policy optimization, ICML  2019    



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
## MC Dropout on latent space models
1. On Uncertainty in Deep State Space Models for ModelBased Reinforcement Learning, TMLR 2022
2. Model-Based Offline Reinforcement Learning With Uncertainty Estimation and Policy Constraint, IEEE TAI 2024

  

# 📘Predictive Propgation during model rollout

# 📘Neat history - check related works in 
1. Modified PETS
2. Trust the Model Where It Trusts Itself- Model-Based Actor-Critic with Uncertainty-Aware Rollout Adaption
3. On Rollouts in Model-Based Reinforcement Learning

# 📘Model Calibration
1. A. P. Dawid. The well-calibrated Bayesian. Journal of the American Statistical Association, 1982
2. M. H. DeGroot and S. E. Fienberg. The comparison and evaluation of forecasters. The statistician, 1983.
3.   Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning, NIPS 2020        
4. Accurate Uncertainties for Deep Learning Using Calibrated Regression, ICML 2018
5. Near-optimal Regret Bounds for Reinforcement Learning,  NIPS 2009 (Error radius is learned from data (concentration inequalities → Hoeffding, Bernstein, GP posteriors))      

# 📘SafeRL
1. Javier Garc´ıa, Fern, and o Fern´andez. A comprehensive survey on safe reinforcement learning. JMLR, 2015.
2. Lukas Brunke, Melissa Greeff, Adam W Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, and Angela P Schoellig. Safe learning in robotics: From learning-based control to safe reinforcement
 learning. Annual Review of Control, Robotics, and Autonomous Systems, 2022.
## Safe exploration in deep RL
1. 



# Constrained MDPs
1.  E. Altman. Constrained Markov Decision Processes. Chapman and Hall, 1999.
## Discrete state-action space
1. Yonathan Efroni, Shie Mannor, and Matteo Pirotta. Exploration-exploitation in constrained mdps.
2. Sharan Vaswani, Lin Yang, and Csaba Szepesvari. Near-optimal sample complexity bounds for constrained mdps. NeurIPS, 2022
3. Dongsheng Ding, Kaiqing Zhang, Jiali Duan, Tamer Bas¸ar, and Mihailo R Jovanovi´c. Convergence and sample complexity of natural policy gradient primal-dual methods for constrained mdps.
4. Adrian M¨uller, Pragnya Alatur, Volkan Cevher, Giorgia Ramponi, and Niao He. Truly no-regret learning in constrained mdps.
5. Exploration-exploitation in constrained mdps, 2020  
## Continuous state-action space
### Deep model based approach (pros: sample efficient, cons: need to learn well calibrated model)
1. Safe Model-based Reinforcement Learning with Stability Guarantees, NIPS 2017
2. ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning, ICLR 2025
3. SafeDreamer: Safe Reinforcement Learning with World Models, ICLR 2024
### Lagrangian based approach  (pros: , cons: Lagrange multiplier may not work well in practice due to oscillations and overshoot)
1. Constrained policy optimization. ICML, 2017
2. Saut´e rl: Almost surely safe reinforcement learning using state augmentation, ICML, 2022.
3. Responsive safety in reinforcement learning by PID lagrangian methods. In ICML, 2020.
#### SAC-Lagrangian (CMDP)
1. Learning to Walk in the Real World with Minimal Human Effort, CORL 2020
2. WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning.AAAI 2021.
3. Safe off-policy deep reinforcement learning algorithm for volt-var control in power distribution systems. IEEE Transactions on Smart Grid 2019
4. CONSERVATIVE SAFETY CRITICS FOR EXPLORATION, ICLR 2021

     

### Penalty based approaches (pros: Simple, stable, strong constraint sastisfcation, cons: Leads to suboptimal policies)
1. Ipo: Interior-point policy optimization under constraints, AAAI 2020
2. Penalized proximal policy optimization for safe reinforcement learning,
3. Log barriers for safe black-box optimization with application to safe reinforcement learning, JMLR 2024
4. Constrained reinforcement learning with smoothed log barrier function,
5. P2bpo: Permeable penalty barrier-based policy optimization for safe rl, AAAI 2024
### Trust region based methods (pros: , cons: Oscillates around the constraint boundary with high overshoot)    
1.  Constrained policy optimization. ICML, 2017
2.  Projection-based constrained policy optimization
3.  Embedding Safety into RL: A New Take on Trust Region Methods, ICML 2025

# 📘Gradient free optimisation (popultion based)
1. The Cross-Entropy Method for Combinatorial and Continuous Optimization, Methodology and Computing in Applied Probability, 1999
2. The Cross-Entropy Method for Optimization. In Handbook of Statistics.
3. Constrained cross-entropy method for safe reinforcement learning. NIPS 2018
## CEM based SafeRL
1. SafeDreamer — Safe Reinforcement Learning with World Models, ICLR 2024  
2. Safe Planning and Policy Optimization via World Model Learning (SPOWL), arXiv 2025  
3. Safe Reinforcement Learning with Model Uncertainty Estimates, arXiv 2018  
4. Exploring Under Constraints with Model-Based Actor-Critic and Safety Filters,CoRL 2024 (PMLR 2025)      
5. Look Before You Leap: Safe Model-Based Reinforcement Learning with Human Intervention (MBHI), CoRL 2022
6. Constrained Model-based Reinforcement Learning with Robust Cross-Entropy Method, arXiv 2020
7. Safe Reinforcement Learning with Minimal Supervision, arXiv 2025  
8. Constrained Cross-Entropy Method for Safe Reinforcement Learning, NeurIPS 2018  
9. Data-Efficient Safe Reinforcement Learning Algorithm, ALA Workshop @ AAMAS 2022  


# 📘Exploration
1. [Exploration and Exploitation-10703 Deep Reinforcement Learning	and	Control](https://www.cs.cmu.edu/~rsalakhu/10703/Lectures/Lecture_Exploration.pdf)    
## Naïve random exploration (best suited for tabular data, compute costly for large dimensional problem)
1. ε-greedy → Watkins (1989), Sutton & Barto (1998/2018).
2. Softmax/Boltzmann → Sutton & Barto, Kaelbling et al. (1996), Thrun (1992).
## Directed exploration- Uncertainty-driven exploration focuses exploration where where the agent’s knowledge is ambiguous.
1. Optimism in the Face of Uncertainty (OFU)    
2. Posterior (Thompson) Sampling
3. Intrinsic Motivation / Bonus-Based
## Optimistic exploration papers (Areas to explore: Computational tractability of OFU, combining OFU with safety guarantees)
1. OFU -  When uncertain about the environment, act as if the most optimistic plausible model (consistent with observed data) is true.    
2. [The need for Explicit Exploration in Model-based Reinforcement Learning](https://berkenkamp.me/blog/2020-12-06-mbrl-exploration/#Mania2019Certainty)
3. H-UCRL - Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning, NIPS 2020
4. UCRL - Near-optimal Regret Bounds for Reinforcement Learning, JMLR 2010 (Tabular MDPS)
5. R-Max - A General Polynomial Time Algorithm for Near-Optimal Reinforcement Learning, JMLR 2023 (Tabular MDPS)
6. GP-UCRL - Online Learning in Kernelized Markov Decision Processes (continuous space)      
7. Regret Bounds for the Adaptive Control of Linear Quadratic Systems, JMLR 2011    
8. Trust-region UCRL meta-algorithm (SLBO) - Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees, ICLR 2019
9. Optimism-driven exploration for nonlinear systems - ICRA, 2015 (First extension of UCRL-style optimism to continuous nonlinear systems via generalized linear models.)
10. Safe Exploration in Reinforcement Learning: Theory and Applications in Robotics
11. The Many Faces of Optimism: a Unifying Approach, ICML 2008    
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

## OFU for linear models (basics)
1. In classical Optimism in the Face of Uncertainty (OFU) (like in tabular RL or linear models), you maintain confidence sets around your estimated model (transition probabilities or dynamics).
2. Then you pick the “most optimistic” model in that set — i.e., the one that maximizes the reward estimate but is still plausible.
3. This requires explicit uncertainty quantification (e.g., concentration inequalities, confidence ellipsoids).
## OFU for continuous non-linear/high dimensional models (basics)
* For nonlinear dynamics (like neural networks as dynamics models), explicit uncertainty quantification is intractable:      
* You can’t get simple closed-form confidence sets. Bounds like Hoeffding/Azuma are too loose. Deep models don’t behave linearly, so confidence regions are not ellipsoids but highly irregular.
  
## OFU for CMDP - continuous non-linear/high dimensional models 
1. ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning, ICLR 2025
2. Constrained Policy Optimization via Bayesian World Models, ICLR 2022

   
# 📘Bootstrap sampling    
1. [Bootstrap resampling](https://towardsdatascience.com/bootstrap-resampling-2b453bb036ec/)


# 📘Key papers
1. Efficient Model-Based Reinforcement Learning through Optimistic Thompson Sampling (HOT-GP), ICLR 2025    
2. Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning (H-UCRL), NIPS 2020    
3. Optimism-Driven Exploration for Nonlinear Systems, ICRA 2015 (Extends OFU for nonlinear continuous systems)
4. Combining Pessimism with Optimism for Robust and Efficient Model-Based Deep Reinforcement Learning (RH-UCRL), ICML 2021
5. DOPE: Doubly Optimistic and Pessimistic Exploration for Safe Reinforcement Learning, NIPS 2022

# 📘Other papers
1. A Unified View on Solving Objective Mismatch in Model Based Reinforcement Learning, 2024
2. Low Level Control of a Quadrotor with Deep Model-Based Reinforcement Learning, RA-L 2019 (Nice low level implementation)
3. A Unified View on Solving Objective Mismatch in Model Based Reinforcement Learning


# 📘Dreamer world model based papers
1. Mastering diverse control tasks through world models, Nature 2025
2. Daydreamer: World models for physical robot learning, CORL 2023
3. Mastering Atari With Discrete World Models, ICLR 2021
4. Dream To Control: Learning Behaviours by Latent Imagination, ICLR 2020
 


# 📘GP model based control papers 
 1. Gaussian processes for dynamics learning in model predictive control, ARC 2025  

   


# 📘Benchmark Analysis
1. Benchmarking Deep Reinforcement Learning for Continuous Control, ICML 2016    
2. Reinforcement Learning with Deep Energy-Based Policies, ICML 2017    

# 📘Metrics
1. Deep Reinforcement Learning at the Edge of the Statistical Precipice, NIPS 2021
2. [Blog](https://agarwl.github.io/rliable/)
3. [Google AI](https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/)    

 
# 📘Toolbox
1. [MBRL-Lib Meta AI](https://github.com/facebookresearch/mbrl-lib). [Paper : MBRL-Lib: A Modular Library forModel-based Reinforcement Learning](https://arxiv.org/pdf/2104.10159)
2. [DI-engine](https://github.com/opendilab/DI-engine)
3. [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
4. [Torch RL](https://docs.pytorch.org/rl/stable/index.html#)
5. [GPytorch](https://gpytorch.ai/)

# 📘Debugging tips
1. [Debugging Deep Model-based Reinforcement Learning Systems](https://natolambert.com/writing/debugging-mbrl)
2. [Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)
3. [RL debugging advice](https://github.com/andyljones/reinforcement-learning-discord-wiki/wiki#debugging-advice)
4. 

# 📘TO-DO
1. [Check quan vuong work](https://github.com/quanvuong/paper_summaries/tree/master)





