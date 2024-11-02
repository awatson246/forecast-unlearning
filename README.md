# Time-Series Tune-Up: Securing Time-Series Models with Unlearning

## Project Definition
As machine learning models drive more decisions in day-to-day life, the ethical implications of these models and their training data have become a growing topic of conversation. Among many concerns, the consequences of reverse-engineering algorithms to uncover sensitive data have been a primary driver behind the model unlearning movement [1]. Although current model unlearning strategies have been primarily concentrated on image and text data [2], time-series reverse-engineering poses a great risk to a variety of domains including healthcare, networking, and finance. As more of these industries use time series data to detect anomalous heartbeat patterns, monitor site traffic for attacks, and predict stock markets, the risks of model reverse-engineering grow. 

In 2018, the fitness tracking app Strava inadvertently revealed the locations of military bases through its publicly available heatmap [3]. The time-series data collected from users’ activities was used to reconstruct sensitive information, exposing the locations and movements of military personnel. Although this incident was not directly caused by model reverse-engineering, it highlighted growing concerns regarding how time-series GPS data may be used maliciously [4]. Companies like Strava have changed their user interface to address some of these concerns. However, user's private data still fuels algorithms making predictions and accounting for GPS errors [5]. Thus, despite policy changes to address known issues, time-series algorithms are still at great risk of exposing private information to bad actors. 

This project will explore the concept of unlearning sensitive GPS data used for time series forecasting to address the feasibility of unlearning information used in time-series algorithms. 

## Existing Methods and Limitations
### Existing Research 
Model unlearning is an emerging field that concentrates on the ability of models to forget or remove specific data. Existing research addresses concerns related to data privacy, system updates, and model robustness [6]. Researcher's aim to prove that trained models can eliminate the influence of certain data points without retraining, which may be computationally expensive and time-consuming.

The majority of model unlearning strategies are predominantly focused on image and text data [2]. There are, however, a handful of studies that have fixated on time-series data algorithms which pose unique problems and concerns. For example: 
* Ye and Lu (2024) propose an unlearning strategy for sequential unlearning systems to privatize these systems and remove "specific client information" [7].
* Du, Chen, et al. (2019) present an unlearning method for anomaly detection algorithms to improve performance, particularly in how systems approach false positives and false negatives [8].
* Fan(2023) proposes a model unlearning technique specifically oriented towards IoT (Internet of Things) anomaly detection models to address growing industrial security concerns [9].

### Limitations
While there have been many developments in the field of unlearning, many limitations still exist: 
* While a few studies focused on time-series models exist, they represent a minority of the existing research despite the prevalence of the models.
* Comparable studies generally address anomaly detection, which is just one of many time-series-centered models. 
    \item Some papers identify problems with exploding loss and catastrophic forgetting in the unlearning process [8]}.
* Scalability has been a universal concern for model unlearning, as the process historically does not scale well for models trained on large datasets [10].

## Citations
[1] H. Liu, P. Xiong, T. Zhu, and P. S. Yu, “A Survey on Machine Unlearning: Techniques and New Emerged Privacy Risks,” Jun. 2024, arXiv:2406.06186 [cs]. [Online]. Available: http://arxiv.org/abs/2406.06186

[2] W. Wang, Z. Tian, C. Zhang, and S. Yu, “Machine Unlearning: A Comprehensive Survey,” Jul. 2024, arXiv:2405.07406 [cs]. [Online]. Available: http://arxiv.org/abs/2405.07406

[3] “Data from fitness app Strava highlights locations of soldiers, U.S. bases - CBS News.” [Online]. Available: https://www.cbsnews.com/news/fitness-devices-soldiers-sensitive-military-bases-location-report/

[4] K. Childs, D. Nolting, and A. Das, “Heat Marks the Spot: De-Anonymizing Users’ Geographical Data on the Strava Heatmap.”

[5] “Moving Time, Speed, and Pace Calculations,” Oct. 2023. [Online]. Available: https://support.strava.com/hc/en-us/articles/115001188684-Moving-Time-Speed-and-Pace Calculations

[6] Y. Qu, X. Yuan, M. Ding, W. Ni, T. Rakotoarivelo, and D. Smith, “Learn to Unlearn: A Survey on Machine Unlearning,” Oct. 2023, arXiv:2305.07512 [cs]. [Online]. Available: http://arxiv.org/abs/2305.07512

[7] S. Ye and J. Lu, “Sequence Unlearning for Sequential Recommender Systems,” in AI 2023: Advances in Artificial Intelligence, T. Liu, G. Webb, L. Yue, and D. Wang, Eds. Singapore: Springer Nature, 2024, pp. 403–415.

[8] M. Du, Z. Chen, C. Liu, R. Oak, and D. Song, “Lifelong Anomaly Detection Through Unlearning,” in Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, ser. CCS’19. New York, NY, USA: Association for Computing Machinery, Nov. 2019, pp. 1283–1297. [Online]. Available: \url[https://dl.acm.org/doi/10.1145/3319535.3363226]

[9] Jiamin Fan, “Machine Learning and Unlearning for IoT Anomaly Detection,” Dissertation, University of Victoria, 2023.

[10] J. Xu, Z. Wu, C. Wang, and X. Jia, “Machine Unlearning: Solutions and Challenges,” IEEE
Transactions on Emerging Topics in Computational Intelligence, vol. 8, no. 3, pp. 2150–2168, Jun. 2024, conference Name: IEEE Transactions on Emerging Topics in Computational Intelligence. [Online]. Available: \[https://ieeexplore.ieee.org/abstract/document/10488864]
