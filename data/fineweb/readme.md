
## openwebtext dataset

after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens (9,035,582,198)
- val has ~4M tokens (4,434,897)

this came from 8,013,769 documents in total.

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset

C:\Users\fzkuj\anaconda3\envs\memory\python.exe "C:\Users\fzkuj\PycharmProjects\memoGPT\data\ fineweb\prepare.py" 
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
loading the dataset
tokenizing the splits
tokenizing the splits (num_proc=8):   0%|          | 0/9667264 [00:00<?, ? examples/s]Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
tokenizing the splits (num_proc=8):   0%|          | 1921/9667264 [00:15<5:05:29, 527.32 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (60232 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 2467/9667264 [00:16<3:01:28, 887.63 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (40556 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 10174/9667264 [00:20<1:17:39, 2072.39 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (97341 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 12152/9667264 [00:21<1:07:07, 2397.02 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (37023 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 18018/9667264 [00:26<1:20:38, 1994.15 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (35664 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 22820/9667264 [00:28<1:42:29, 1568.35 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (88928 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 23389/9667264 [00:28<1:40:13, 1603.83 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (46474 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):   0%|          | 41574/9667264 [00:42<1:58:00, 1359.53 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (37565 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8): 100%|██████████| 9667264/9667264 [1:46:03<00:00, 1519.28 examples/s]
tokenizing the splits (num_proc=8):   0%|          | 0/4837 [00:00<?, ? examples/s]Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
tokenizing the splits (num_proc=8):  15%|█▍        | 721/4837 [00:17<00:47, 87.02 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (39698 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8):  68%|██████▊   | 3287/4837 [00:18<00:00, 1562.03 examples/s]Token indices sequence length is longer than the specified maximum sequence length for this model (42851 > 32768). Running this sequence through the model will result in indexing errors
tokenizing the splits (num_proc=8): 100%|██████████| 4837/4837 [00:21<00:00, 229.02 examples/s]
writing C:\Users\fzkuj\PycharmProjects\memoGPT\data\ fineweb\train.bin: 100%|██████████| 1024/1024 [03:23<00:00,  5.04it/s]
writing C:\Users\fzkuj\PycharmProjects\memoGPT\data\ fineweb\val.bin: 100%|██████████| 1024/1024 [00:02<00:00, 506.84it/s]

Process finished with exit code 0