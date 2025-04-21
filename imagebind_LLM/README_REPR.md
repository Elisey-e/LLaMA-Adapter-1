Currently, Image-Bind could not be started, there is a problem with the cuda drivers. The inference was conducted only for the LLama Adapter and other models.


## Setup

* Setup up a new conda env. Install ImageBind and other necessary packages.
  ```bash
  # create venv
  python3 -m venv .venv
  source .venv/bin/activate
  # install ImageBind
  cd ImageBind
  pip install -r requirements.txt
  # install other dependencies
  cd ../
  pip install -r requirements.txt
  pip install -r req_fix.txt

  python3 demo.py
  ```

