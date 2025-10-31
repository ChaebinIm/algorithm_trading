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
<img width="699" height="145" alt="image" src="https://github.com/user-attachments/assets/1a04f57d-0d39-4a75-a47a-0ba93100423a" />

- 로컬 작업 중 사소한 커밋. 커밋은 자주해줄수록 좋음. (버전관리 가능)
<img width="694" height="169" alt="image" src="https://github.com/user-attachments/assets/c797e156-dc04-417c-8a28-ead304cadeef" />


- 다른 컴퓨터에서 이어서 작업하고 싶을 때
<img width="709" height="569" alt="image" src="https://github.com/user-attachments/assets/485b7a8e-b6f6-4d7a-b01d-bbc8631a1351" />

<img width="708" height="199" alt="image" src="https://github.com/user-attachments/assets/9db65936-2614-45aa-9f68-7bc6f5d91f56" />

- 작업 끝났을 때 (pull Request)
<img width="677" height="163" alt="image" src="https://github.com/user-attachments/assets/00f9261c-f7da-445f-81a4-f7c90a3e1b6b" />



