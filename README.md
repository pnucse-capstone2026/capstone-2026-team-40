### 1. 프로젝트 배경

우리나라는 삼면이 바다로 둘러싸여 있으며 항만, 해수욕장, 연안 시설 등 넓은 해양·해안 영역에 대한 지속적인 감시가 요구된다. 특히 최근 소형 보트나 제트스키 등을 이용하여 해상으로 접근하거나 밀입국을 시도한 사례가 발생하면서, 기존의 선박뿐만 아니라 사람, 소형 수상기구, 드론 등 다양한 침투 가능 객체를 신속하게 탐지할 수 있는 감시 기술의 필요성이 증가하고 있다.

그러나 해양과 해안은 감시 범위가 넓고 관측 환경이 지속적으로 변화하기 때문에 영상 기반 감시에 어려움이 있다.따라서 영상 내 객체의 위치와 종류를 자동으로 식별하고, 탐지 결과를 관제자가 직관적으로 확인할 수 있도록 지원하는 AI 기반 객체 탐지 기술이 필요하다.

### 2. 프로젝트 목표

| 구분 | 세부 목표 |
|---------|-----|
| 객체 탐지 | YOLO26n으로 big, middle, small, drone, human의 위치와 클래스를 탐지한다 |
| 함정 데이터 구축 | 해양경찰청 함정도감을 기준으로 대형·중형·소형 경비함정과 특수함정의 실제 자료를 수집한다. | 
| 가상 데이터 구축 | Unity에서 드론과 사람을 다양한 거리·각도·기상 조건으로 생성하여 실제 데이터의 부족한 조건을 보완한다 |
| 함정 세부 분류 확장 | big·middle·small로 탐지된 함정 Crop을 별도 분류 단계에 연결하여 세부 유형 분류로 확장한다 |
| 환경 강건성 평가 | Clear, Cloudy, Rain, Fog, Night 조건의 장면을 구성하여 탐지 양상을 확인한다 |
| 결과 시각화 | Bounding Box, 클래스명, 신뢰도와 세부 분류 결과를 표시하고 간단한 관제형 UI에서 영상을 확인할 수 있도록 한다 |


### 3. 왜 YOLO26인가

본 연구에서는 RF-DETR, YOLO11, YOLO26 계열 모델의 검증 결과를 비교하였다. 현재 확보된 결과에서는 전체 mAP50과 human 클래스 mAP50을 공통 비교 지표로 확인할 수 있으며, 동일 장치에서의 FPS나 지연시간은 측정 자료가 없어 속도 우위를 정량적으로 비교하지 않았다. 최종 시스템에는 YOLO26n을 적용하였다.

•	RF-DETR: 검증 결과 전체 mAP50 97.0%, human 클래스 88.0%를 기록하여 세 후보 중 가장 높은 정확도를 보였다. 다만 현재 자료에는 동일 장치 기준의 추론 속도 수치가 포함되어 있지 않으므로, 본 보고서에서는 RF-DETR의 실시간 처리 성능을 수치로 단정하지 않는다.

•	YOLO11: 검증 결과 전체 mAP50 90.0%, human 클래스 55.0%를 기록하였다. 전체 성능은 최종 YOLO26n과 동일한 90.0%였으나, human 클래스 성능은 YOLO26n보다 낮게 나타났다.

•	YOLO26n: 검증 결과 전체 mAP50 90.0%, human 클래스 58.0%를 기록하였다. 전체 mAP50은 YOLO11과 동일했으며, human 클래스에서는 YOLO11 대비 3%p 높은 결과를 확인하였다.


### 4. 시스템 설계

| 단계 | 입력 | 처리 | 출력 |
|----|----|----|----|
| 데이터 구축 | 웹 이미지, YOUTUBE 영상, Unity 이미지 | 수집·생성·정제·중복 제거 | 원본 및 정제 데이터 |
| 라벨링 | 정제 이미지 | Roboflow Bounding Box 작성 및 검수	 | 이미지와 txt 라벨 |
| 1단계 탐지 | 이미지/영상 프레임 | YOLO26n 추론 | 	big·middle·small·drone·human, confidence, Bounding Box |
| 세부 분류 확장 | big·middle·small 함정 Bounding Box Crop | 별도 분류 단계 | 함정 세부 유형 또는 unknown |
| 시각화 | 탐지·분류 결과 | 영상 결과 결합 및 UI 출력 |	탐지 영상과 요약 정보 |


![인터페이스 이미지]([https://user-images.githubusercontent.com/100384365/192478661-5dc79a18-b076-48ef-b842-bcf65b0d8d44.jpg](https://github.com/pnucse-capstone2026/capstone-2026-team-40/blob/42b526cb95799b6bb8e61b737bd05b534f9eff73/docs/04.%EC%9D%B4%EB%AF%B8%EC%A7%80%ED%8C%8C%EC%9D%BC/main_interface.png))

### 5. 데이터 구축

본 연구는 데이터의 출처와 특성에 따라 함정 실사 데이터와 Unity 합성 데이터를 구분하여 확보한다.

함정은 실제 외형을 반영하기 위해 해양경찰청 자료, 공개 이미지와 영상을 중심으로 수집하고, 드론과 사람은 원하는 거리·각도·기상 조건을 반복적으로 구성할 수 있는 Unity 데이터를 활용한다.

이후 데이터를 정제하고 Roboflow에서 big, middle, small, drone, human의 최종 탐지 클래스 체계로 라벨링하여 YOLO26n 학습에 사용한다.

| 대상 | 주요 데이터 출처 | 구축 방법 | 활용 목적 |
|-----|-----------------|------------|----------|
| 함정 |	해양경찰청 함정도감 | Google 이미지, YouTube 영상	실제 이미지 수집 및 영상 프레임 추출 | 실제 함정 외형과 다양한 시점 학습 |
| 드론 | Unity 3D 모델 및 가상환경 | 거리·각도·기상별 합성 이미지 생성 | 해양 배경에서 부족한 드론 데이터 보완 |
| 사람 | Unity 3D 모델 및 가상환경 | 부두·해안 배경에서 위치·거리 변화 | 해안 접근 상황의 사람 데이터 보완


### 6. 연구 결과 분석

아래의 평가항목을 기준으로 분석하였다.
| 평가 항목 |	확인 결과 |
|:---------:|----------|
| YOLO26n 전체 검증 성능 | Validation Set: mAP@50 90.0%, Precision 92.9%, Recall 87.7%, F1 90.2% |
| 클래스별 | AP@50	big 100%, middle 100%, small 99%, drone 93%, human 58% |
| 후보 모델 비교 | 전체 mAP50: RF-DETR 97.0%, YOLO11 90.0%, YOLO26n 90.0% / human mAP50: 88.0%, 55.0%, 58.0% |
| 환경별 시나리오 | Clear, Cloudy, Rain, Fog, Night 장면을 구성하여 정성적으로 확인 | 
| 함정 세부 분류 | 함정 Crop 기반 세부 유형 분류 구조를 구성했으나 확정 모델명·정량 성능 수치는 현재 자료에 없어 미기재 |
| 미산출 정량 항목 | mAP@0.5:0.95, 동일 장치 추론 시간/FPS, 기상 조건별 mAP는 현재 확보 자료에 수치가 없어 미기재 |

### 7. 팀 구성

| 성명 | 담당 역할 | 주요 수행 내용 |
|:----:|:--------:|----------------|
| 김성윤 | 라벨링·YOLO26·함정 분류 구조 | Roboflow Bounding Box 라벨링 및 검수, 최종 클래스 구조 정리, YOLO26 학습·재학습과 성능평가, 오탐·미탐 분석, 함정 Crop 기반 세부 분류 구조 구성 |
| 김윤지 | 함정 데이터·최종 시각화·문서화 | 	해양경찰청 함정도감 기반 실사 데이터 수집·정리, Google/YouTube 자료 조사와 영상 프레임 추출, 최종 모델 탐지 결과 시각화, Streamlit 관제형 UI 구성, 최종보고서 및 포스터 제작 |
| 서연우 | Unity 합성 데이터 제작 | 포토그래메트리 기반 연안 가상환경 구성, 드론·사람 3D 모델 적용, Clear/Cloudy/Rain/Fog/Night Scene 구성, 거리·각도·위치별 합성 이미지 생성, 자동 캡처 및 파일명 관리, 합성 데이터 품질 검토 |

### 8. 참고 문헌

[1] J. H. Kim, N. Kim, Y. W. Park, and C. S. Won,
“Object Detection and Classification Based on YOLO-V5 with Improved Maritime Dataset,”
Journal of Marine Science and Engineering, vol. 10, no. 3, p. 377, 2022. DOI: 10.3390/jmse10030377.

[2] 해양경찰청, 「함정·항공기 도감」, 대한민국 국회도서관 소장자료.

[3] L. A. Varga, B. Kiefer, M. Messmer, and A. Zell,
“SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water,” Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2260–2270, 2022.

[4] B. He, X. Li, B. Huang, E. Gu, W. Guo, and L. Wu,
“UnityShip: A Large-Scale Synthetic Dataset for Ship Recognition in Aerial Images,” Remote Sensing, vol. 13, no. 24, p. 4999, 2021.

[5] D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabally, and C. Quek,
“Video Processing From Electro-Optical Sensors for Object Detection and Tracking in a Maritime Environment: A Survey,” IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 8, pp. 1993–2016, 2017.

[6] Roboflow, “YOLO26 - Object Detection,” Inference Models Documentation, accessed 2026-09-07.
URL: https://inference-models.roboflow.com/models/yolo26-object-detection/


### 9. 소개 및 시연 영상

[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://youtu.be/1d5YcEO1V1U?si=tFSh4Ua90-TlaHVz)
