A simple NER training task with an invoice extraction dataset.

# Data, library preparation
Download the Multi-layout Invoice Document Dataset (MIDD) for exmaple from this link, and put `IOB` to `data` directory
```commandline
pip install -r requirements.txt
```
# Train the model
```commandline
python train.py
```

# NER on test data
```commandline
python ner_test.py
```

You will get results like this. 
![Alt text](image/NER_result_example.png?raw=true "Overview")
