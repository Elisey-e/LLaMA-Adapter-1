Currently, Image-Bind could not be started, there is a problem with the cuda drivers. The inference was conducted only for the LLama Adapter and other models.


## Setup

* Setup up a new conda env. Install ImageBind and other necessary packages.
  ```bash
  # create venv
  python3 -m venv .venv
  source .venv/bin/activate

  pip install -r requirements.txt
  # install other dependencies
  cd ../
  pip install -r requirements.txt
  pip install -r req_fix.txt

  python3 demo.py
  ```

## Measurements

### Общие положения

Все измерения находятся в папке datasets.
..._metrics.json - рассчет статистик инференса
..._results.json - разметка полученная в ходе инференса


### old_measurements
результаты, не вошедшие в папки с датасетами, имели значение в начале для выбора используемой авторами метрики: case_insensitive_accuracy, word_accuracy, strict_word_accuracy, small_lexicon_accuracy, ...
Послужили причиной выбора указанной в презентации реализации word accuracy.

