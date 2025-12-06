# AGGA-MVFLN: Multivariate Time Series Forecasting via Adaptive Generalized Graph Accompanied with Multi-View Learning in Frequency Domain (ICMR2025)

## Introduction

A growing body of recent researches have migrated graph structure learning (GSL) to the multivariate time series forecasting (MTSF), which lays the foundation for the promotion of ''Generalized Graph'' for multimedia MTSF applications. In other words, we expect generalized graph to encompass the learning of inter-variable, inter-temporal and latent correlations, becoming a universal tool for multivariate correlations learning. However, due to the heterogeneity of multivariate time series in distribution, graph learning inevitably captures inaccurate relationships, which requires the quality of graph learning; Meanwhile MTSF often requires instant predictions for decision-making in real-world, which also challenges the speed of GSL. To solve these challenges, we propose AGGA-MVFLN, namely Adaptive Generalized Graph Accompanied Multi-View Frequency Learning Network. Specifically, we introduce an adaptive generalized graph structure from multi-view (global and local) to capture diverse ''spatio-temporal patterns''. Subsequently, we utilize the Fast Fourier Transform to map them into the frequency domain, and enhance the quality of the generalized graph by collaboratively learning the complementarities and differences through reconstructed ''spatio-temporal patterns'' and error-driven supervised training of adaptive graph. The benefits are: (1) The frequency domain can disentangle complex temporal patterns, making the process of learning multivariate relationships more robust. (2) Multi-view learning can significantly reduce training time by preset and seamless integration (i.e., the multi-task loss form). (3) ''Generalized Graph'' can be regarded as universal component for multivariate correlation learning. Evaluation of 9 real-world datasets confirms the superiority of AGGA-MVFLN over SOTA benchmark.



## running programme

### Single-process experiment: running exp.py

> >Running style


> >(1) Setting up the experimental task environment: you can do a manual setup of parser.add_argument in exp.py

> >1.1 "model_name":"agga-fln"

> >1.2 "dataset_name": The corresponding "help" in exp.py selects the dataset.

> >1.3 "inp_len": 96

> >1.4 "pred_len": 96/192/336/720

> >(2) Run it directly from the command line：nohup python -u exp.py > train.log 2&>1 &

> >(3) No pre-setting, run directly from the command line：

> > for example：nohup python -u exp.py --note "agga-fln-weather-96" --model_name "agga-fln" --dataset_name "weather" --inp_len 96 --pred_len 96 > train.log 2>&1 &

> > The results are in the corresponding train.log file.
## Citation

If you find this repo helpful, please cite our paper. 

```
 @inproceedings{lei2025agga,
  title={AGGA-MVFLN: Multivariate Time Series Forecasting via Adaptive Generalized Graph Accompanied with Multi-View Learning in Frequency Domain},
  author={Lei, Jierui and Chen, Fangzheng and Tang, Haina},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={653--661},
  year={2025}
}
```
"""

