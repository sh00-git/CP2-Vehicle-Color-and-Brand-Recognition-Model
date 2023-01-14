## CP2 : 차량 색상 분류 및 브랜드 인식 모델

- 개발기간
    - 2022.11.18 ~ 2022.12.14
- 사용 언어 및 라이브러리
    - `Python`, `OpenCV`, `pillow`, `Pytorch`, `sklearn`, `pandas`, `matplot`, `seaborn`

## 💡 Topic

- EfficientNet과 YOLO 모델을 사용하여 차량의 색상을 분류하는 시스템과 차량의 브랜드를 인식하는 모델을 구현하는 프로젝트
- AI 부트캠프 CP2

## ❓ 문제정의

본 프로젝트는 기업 협업 프로젝트로 차량의 색상을 분류하는 모델을 구현하며 AI HUB 데이터 구조를 이해하고 최종적으로 차량의 브랜드를 인식하는 모델을 구현하는 프로젝트이다.

## 🔍 프로젝트 진행과정

<img src="https://user-images.githubusercontent.com/76083173/212466660-cbb70f59-4540-4f20-bc91-14518cd418b4.png" height="300" />

## 📚 데이터 셋

[AI-Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=554)

- AI HUB의 차량 외관 영상 데이터
- 차량 외관과 14개 파트 정보가 담긴 json, jpg 데이터
- 1920 x 1080 크기의 322,664장

## ✍🏻 프로젝트 수행 과정 및 결과

1. 데이터 확인
    - 기업에서 제공받은 코드를 통해 라벨링 데이터를 labelme 포맷으로 변경하여 이미지 시각화를 통해 바운딩박스 확인
    - 바운딩박스 중 차량 전체라는 label이 별도로 존재하는 것을 확인
    
      <img src="https://user-images.githubusercontent.com/76083173/212466695-a76d2aa7-27c8-4446-a45c-9f9a72e68f29.png" height="260" />
      <img src="https://user-images.githubusercontent.com/76083173/212466756-537a400a-6128-4be8-a041-84712172db63.png" height="260" />
    
2. 차량 색상 분류 알고리즘 구현
    - 데이터 셋 구축
        - 차량 전체 모습이 담긴 이미지만 크롭하여 색상 별(검정, 빨강, 파랑, 회색, 흰색)로 분류
        - 가장 적은 데이터 수를 가진 빨강색 개수에 맞춰 각 색상 별 3704개씩인 데이터를 train, validation, test set으로 나눔
            - Train : test = 8 : 2
            Train : validation = 9 : 1
    - 모델 선정
        - EfficientNet-B0 사용
            - ImageNet에서 기존 ConvNet보다 8.4배 작으면서 6.1배 빠르고 더 높은 정확도를 가진다는 특징이 있음.
            - 다양한 버전들 중 차량의 색상을 파악하는 단순 학습과정이므로 B0도 적합하다고 판단.
    - 학습 결과
        - 최종 학습 정확도 : 90.8906
        - Test Set 정확도 : 91%
        - 빨강색인 것을 회색으로 회색인 것을 빨강색으로 잘못 예측하는 모습도 보이는 것을 보아 예측 과정에서 어디 부분을 보고 예측하는지 파악하기 위해 히트맵을 통해 확인하는 과정이 필요해 보임.
        - Test Set 평가지표
            
            <img src="https://user-images.githubusercontent.com/76083173/212466770-fb0ee40e-4980-4203-a588-197b71fc9a1f.png" height="230" />
            
        - Confusion matrix
            
            <img src="https://user-images.githubusercontent.com/76083173/212466801-68452545-f7ea-4d3b-989c-d470c538fee9.png" height="300" />
            
        - 예측 결과
            
            <img src="https://user-images.githubusercontent.com/76083173/212466818-b2e74ae6-380d-4c2e-b004-0048748a791a.png" height="450" />
            
3. 차량 브랜드 분류
    - 브랜드 리스트 구축
        - 현대 차 데이터만 사용하여 우선 학습 진행.
        - 클래스와 연식으로 나누어진 65개의 브랜드 리스트 구축
        - YOLO 모델 학습을 위해 obj.names파일로 저장
        
        <img src="https://user-images.githubusercontent.com/76083173/212466852-7a0ba717-ce78-4c2b-aa37-0e7ade18be5f.png" height="200" />
        
    - YOLO 모델에 맞도록 데이터 형식 변경
        - AI HUB 데이터의 바운딩박스 형식
            - 바운딩 박스의 네 꼭지점 좌표
        - YOLO 데이터의 바운딩박스 형식
            - class, x, y, w,h
            - x, y : 바운딩박스 중심점, 그리드 셀의 범위에 대한 상대값 입력
            - w, h : 전체 이미지대비 바운딩 박스의 상대값(너비, 높이)
    - 차량 방향 탐지
        - 기업에서 제공받은 학습 모델 사용
        - 이미지에서 인식된 차량의 방향을 탐색하여 반환하는 모델
        - 목표 차량의 방향만 탐지하기 위해 이미지를 크롭한 데이터를 input으로 넣어준 후 방향 탐지가 안된 1320개의 데이터를 제외하고 방향별 데이터 셋 구축
    - 모델 학습
        - 차량 앞모습 데이터를 사용하여 모델 학습 진행
        - YOLOv4 학습을 위한 cfg 파일 설정
            - 총 class 개수는 65개
        - 약 1300 Iteration 학습 후 결과(1/100 진행)
            - 실제 label은 현대 아이오닉 2018 이지만 0.57%의 확률로 현대 쏘나타 2020으로 예측. Bbox가 이상한 곳을 가리키고 있음.
                
                <img src="https://user-images.githubusercontent.com/76083173/212466873-10f6b8ec-357e-42c6-a2a0-da0c01f2f05f.png" height="300" />
                
        - 약 13000 Iteration 학습 후 결과(1/10 진행)
            - Iteration이 거듭될수록 loss값이 작아지고 maP값이 증가하는 모습을 보임
                
                <img src="https://user-images.githubusercontent.com/76083173/212466921-dfa60722-68c7-478d-8392-a53c9de807cf.png" height="500" />
                
            - maP값이 64.4%일 때 테스트 이미지를 예측한 모습
                - 실제 label: 현대 쏘나타 2017
                - 예측 label: 현대 쏘나타 2017
                - 1300 Iteration 진행했을 때와는 다르게 Bbox도 제대로 그리고 있음.
                
                <img src="https://user-images.githubusercontent.com/76083173/212466949-22e80309-f950-4a02-94bc-03ed5c2d6773.png" height="300" />
                

## ✨ Learned

- 학습시키고자 하는 모델에 맞게 `데이터셋을 구축`하는 과정을 수행할 수 있었다.
- `YOLO 모델에 맞게 데이터를 전처리`하는 과정을 수행할 수 있었다.
- `YOLOv4`의 학습 과정을 수행할 수 있었다.
- 학습시킨 셀이 종료되었다고 해서 학습이 완료됨을 의미하지 않는다는 것을 알게 되었다.
- 기업과 미팅을 진행하면서 모델을 돌리고 결과를 보는 것도 중요하지만, `모델이 어느 부분을 보고 예측을 하였는지`에 대한 인사이트를 얻는것도 중요하고 히트맵을 통해 그러한 부분을 파악할 수 있다는 점을 배울 수 있었다.
- 구글 드라이브에 데이터를 업로드하는 시간이 오래 걸린다는 것을 체감하였다.
- YOLOv4 학습 시 loss와 maP를 확인할 수 있는 그래프인 chart.png 파일이 제공되지만 셀이 종료되고 마지막 가중치로 학습을 다시 돌릴 때에 chart.png 파일이 초기화 된다는 것을 알게 되었다.
- yolo 학습은 darknet을 통해 진행되지만 학습된 모델을 사용할때에는 openCV를 통해서도 가능하다는 것을 알게되었다.
