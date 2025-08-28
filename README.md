# AGGA-MVFLN: Multivariate Time Series Forecasting via Adaptive Generalized Graph Accompanied with Multi-View Learning in Frequency Domain (ICMR2025)


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

### Please cite this if you like this code:

> > @inproceedings{lei2025agga,
  title={AGGA-MVFLN: Multivariate Time Series Forecasting via Adaptive Generalized Graph Accompanied with Multi-View Learning in Frequency Domain},
  author={Lei, Jierui and Chen, Fangzheng and Tang, Haina},
  booktitle={Proceedings of the 2025 International Conference on Multimedia Retrieval},
  pages={653--661},
  year={2025}
}
