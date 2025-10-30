# algorithm_trading
algorithm trading by Chap. It starts with crypto currency. but, it will be bigger.

# 작업형태
여러 컴퓨터에서 로컬로 작업하되, 완료(또는 중간)시 github에 올리는 구조. 아래와 같이 운영되면 가장 깔끔하고 안전함.

### 기본 원칙
- github이 단 하나의 기준 원본(원격, origin)이고, 각 PC는 클론된 로컬 사본.
- 모든 PC는 같은 리포지토리를 git clone해서 쓰되, 작업은 브랜치 단위로 진행.

### 권장 브랜치 전략(가볍지만 실용적으로)
- main : 항상 배포/공유 가능한 안정 상태
- feature/이름 : 각 작업(기능, 버그픽스)마다 새로 만들고 여기서 커밋.
- 완료되면 PR(Pull Request)로 main에 합치고, 병합 후 feature 브랜치 삭제.
- 혼자서 작업하더라도, PR(Pull Request)를 써두면 변경 이력, 리뷰 메모, 백업이 훨씬 깔끔할 수 있음.

### 각 PC에서의 루틴(가장 중요!!)
- 첫 설정(컴퓨터별로 한번만!)
<img width="709" height="353" alt="image" src="https://github.com/user-attachments/assets/5383f7b8-f260-42f8-becf-40c00ddf42dc" />

- 새 작업을 시작할 때
<img width="696" height="177" alt="image" src="https://github.com/user-attachments/assets/c1b748a1-7bcf-4b85-92f4-452dfcfab305" />

- 로컬 작업 중 사소한 커밋. 커밋은 자주해줄수록 좋음. (버전관리 가능)
<img width="694" height="137" alt="image" src="https://github.com/user-attachments/assets/7870a439-5429-478e-b4a0-da75f96c0db2" />

- 다른 컴퓨터에서 이어서 작업하고 싶을 때
<img width="695" height="197" alt="image" src="https://github.com/user-attachments/assets/2e6bc681-5da2-4059-9672-a46585690848" />

- 작업 끝났을 때 (pull Request)
<img width="700" height="271" alt="image" src="https://github.com/user-attachments/assets/74357e2d-fb70-49bd-9b89-869cb5c6df97" />


