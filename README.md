# 제4회 AI x Bookathon

제 4회 AI x Bookathon 대회에 사용된 코드 입니다. 자세한 후기와 저희 팀이 접근 하였던 방법은 [이 곳](https://cyc9805.github.io/python/project/AIxBookathon-참여후기/)에서 찾아 보실 수 있습니다

# Data Crawling

- **collecting_text.ipynb**
  <br>&emsp; 모델의 전체적인 fine-tuning을 위해서 수집한 [브런치](https://brunch.co.kr/) 데이터의 크롤링에 사용된 코드입니다.

- **collecting_text_for_epsiode.ipynb**
  <br>&emsp; 에피소드 별 키워드를 브런치에서 검색한 후 나타나는 데이터를 크롤링하는 코드입니다.

# Training

&emsp; 모델 훈련은 다음과 같이 진행 하였습니다:
```python
python train.py \
  --data_name 'collected_data.txt' \
  --model_name 'model1' \
  --batchsize 8 \
  --epoch 5 \
  --save_steps 500 \
  --overwrite_ouput_dir True 
  
# Inferring

&emsp; 모델을 통한 다음 문장 생성은 다음과 같이 진행 하였습니다:
```python
python infer.py \
  --sequence '담대한' \
  --model_name 'model1' \
  --maxlen 300 \
  --sample True \
