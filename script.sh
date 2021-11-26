# multi normal
python bert_laban.py train --datatype=semantic 
python bert_laban.py train --datatype=mixatis
python bert_laban.py train --datatype=mixsnips
python bert_laban.py train --datatype=sgd
python bert_laban.py train --datatype=e2e

# single normal
python bert_laban.py train --datatype=atis --data_mode=single
python bert_laban.py train --datatype=snips --data_mode=single

# zero-shot
python bert_laban.py train --datatype=semantic
python bert_laban.py train --datatype=mixatis
python bert_laban.py train --datatype=mixsnips