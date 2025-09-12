## --------------------------------------------------------------------
## 함수이름 : monthly_data
## 매개변수 : 없음
## 반 환 값 : 없음
## --------------------------------------------------------------------
monthly_data={
    "1월": [],
    "2월": [],
    "3월": [],
    "4월": [],
    "5월": [],
    "6월": [],
    "7월": [],
    "8월": [],
    "9월": [],
    "10월": [],
    "11월": [],
    "12월": [],
    "B": ["Back"]
}
def monthly_01():
    for mon, value in monthly_data.items():
        print(mon,value,end='\n' if mon=='B' else ' ')
        print()
    



## --------------------------------------------------------------------
## 함수이름 : printMenu
## 매개변수 : 없음
## 반 환 값 : 없음
## --------------------------------------------------------------------
def printMenu():
    # 메뉴출력
    print('*'*30)
    print('1. Monthly')
    print('2. 종  료')
    print('*'*30)
    
    return input("메뉴 선택 : ")
    

## --------------------------------------------------------------------
## 함수이름 : choice_monthly
## 매개변수 : 없음
## 반 환 값 : 없음
## --------------------------------------------------------------------
# 월별 일정에 인덱스를 추가해서 저장
def choice_monthly():
    
    input_monthly=input("일정을 수정할 월을 입력하세요(예:5월, ..., B) : ".strip())
    
    if input_monthly in monthly_data and input_monthly != 'B':
        print('*'*30)
        print("현재 일정 : ")
        print('*'*30)
        # enumerate : 인덱스와 값을 함께 반환
        for index, item in enumerate(monthly_data[input_monthly], 1):
            print(f"{index}. {item}") # 인덱스 번호와 함께 출력
       
        
        # pirntMenu_02를 호출하여 일정 추가/삭제/종료 선택
        choice=printMenu_02()
        
        if choice == 1:
            content = input(f"{input_monthly}에 추가할 일정을 입력하세요 : ")
        
            # 월별 일정에 추가
            monthly_data[input_monthly].append(content)
        
            print(f"일정이 {input_monthly}에 추가되었습니다 : {content}")
            print(f"{input_monthly}의 일정 : {monthly_data[input_monthly]}")
            
        elif choice== 2:
            try:
                delete_index = int(input(f"{input_monthly}에 삭제할 일정을 입력하세요 : "))
                if 0<= delete_index < len(monthly_data[input_monthly]):
                    deleted_item = monthly_data[input_monthly].pop(delete_index) # 해당 항목 삭제
                    print(f"{deleted_item} 일정 삭제")
                else:
                    print("잘못된 번호를 입력했습니다.")
            except ValueError: # 자료형 불일치하면 
                print("잘못된 입력입니다. 숫자를 입력해 주세요.")
                
            print(f"현재 {input_monthly}의 일정 : {monthly_data[input_monthly]}")           
            
        
        elif choice == 3:
            print("Back")
            return
            
        else:
            print("잘못된 메뉴 선택입니다.")
        
    elif input_monthly=='B':
            print("Back")
            return
        
    else:
            print("잘못된 월 입력입니다. 1에서 12까지의 월을 입력하세요. ")


## --------------------------------------------------------------------
## 함수이름 : printMenu_02
## 매개변수 : 없음
## 반 환 값 : 없음
## --------------------------------------------------------------------
def printMenu_02():
    # 메뉴출력
    print('*'*30)
    print('1. 일정추가')
    print('2. 일정삭제')
    print('3. Back')
    print('*'*30)
    
    choice = input("메뉴 선택 : ")
    
    return int(choice)
