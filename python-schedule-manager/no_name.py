## 기능 함수들 모듈/파일 로딩 -------------------------------
import no_name_func


while True:
    # 메뉴출력 및 선택
    choice=no_name_func.printMenu()
    
    # 종료 조건문
    if choice=='2':
        print('프로그램을 종료합니다.')
        break
    
    # 메뉴에 따른 기능 코드 실행
    elif choice=='1':
        print('*'*30)
        print("< 2025년 >")
        print(no_name_func.monthly_01())
        print('*'*30)
        no_name_func.choice_monthly()
        
    else:
        print("존재하지 않는 메뉴입니다.")
        
        
        





