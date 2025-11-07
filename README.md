# DeepGR4J with extreme flow classification based on Quantile Regression

This repository extends the work from [DeepGR4J](https://github.com/DARE-ML/DeepGR4J) to improve predictions on extreme events. The initial experiments use quantile regression to achieve this. 


### Notebook Overview
 - `quantile_reg_nn.ipynb`: Implements quantile regression for a basic MLP
 - `qdeepgr4j_cnn.ipynb`: Extends this quantile regression workflow to DeepGR4J-CNN model
 - `qdeepgr4j_lstm.ipynb`: Extends this quantile regression workflow to DeepGR4J-LSTM model


### Cite this work
```
@article{kapoor2025qdeepgr4j,
  title={QDeepGR4J: Quantile-based ensemble of deep learning and GR4J hybrid rainfall--runoff models for extreme flow prediction with uncertainty quantification},
  author={Kapoor, Arpit and Chandra, Rohitash},
  journal={Journal of Hydrology},
  pages={134434},
  year={2025},
  publisher={Elsevier}
}
```
 
