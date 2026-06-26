import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import yaml
import threading

# 기존 파이프라인 모듈들 원격 호출
from src.data_pipeline import fetch_raw_data, engineer_features
from src.model_pipeline import train_and_update_model, predict_returns
from src.allocation_pipeline import allocate_assets
from src.backtest_pipeline import run_backtest

class TradingGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML4T 알고리즘 트레이딩 파이프라인 시스템")
        self.root.geometry("650x650")
        self.root.resizable(False, False)
        
        # 설정 정보 로드
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.create_widgets()

    def create_widgets(self):
        # 1. 입력 프레임 (상단)
        input_frame = ttk.LabelFrame(self.root, text=" ⚙️ 투자 설정 입력 ", padding=15)
        input_frame.pack(fill="x", padx=20, pady=15)
        
        # 1-1. 시작 금액 입력 창
        ttk.Label(input_frame, text="시작 금액 입력 (달러):").grid(row=0, column=0, sticky="w", pady=5)
        self.capital_entry = ttk.Entry(input_frame, width=20)
        self.capital_entry.insert(0, str(self.config.get('initial_capital', 10000))) # 기본값 로드
        self.capital_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # 1-2. 리밸런싱 주기 선택 라디오 버튼
        ttk.Label(input_frame, text="리밸런싱 주기 선택:").grid(row=1, column=0, sticky="nw", pady=10)
        
        self.rebalance_var = tk.StringVar(value="21") # 기본값: 1달(21일)
        
        radio_frame = ttk.Frame(input_frame)
        radio_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # 질문자님이 요청하신 5가지 주기 버튼 배치 (영업일 기준 환산)
        periods = [("1일 (매일)", "1"), ("1주일", "5"), ("1달 (월간)", "21"), ("1분기 (분기)", "63"), ("1년 (연간)", "252")]
        for text, value in periods:
            ttk.Radiobutton(radio_frame, text=text, variable=self.rebalance_var, value=value).pack(anchor="w", pady=2)
            
        # 2. 실행 버튼 및 상태바 (중단)
        self.run_btn = ttk.Button(self.root, text="🚀 머신러닝 백테스팅 실행", command=self.start_pipeline_thread)
        self.run_btn.pack(fill="x", padx=20, pady=5)
        
        self.status_label = ttk.Label(self.root, text="대기 중... 설정 입력 후 실행 버튼을 눌러주세요.", foreground="blue")
        self.status_label.pack(pady=5)
        
        # 3. 성과 보고서 표 (하단 Treeview로 표 구현)
        table_frame = ttk.LabelFrame(self.root, text=" 📊 성과 평가지표 비교 성적표 ", padding=15)
        table_frame.pack(fill="both", expand=True, padx=20, pady=15)
        
        # 테이블 컬럼 설정
        self.tree = ttk.Treeview(table_frame, columns=("Metric", "Portfolio", "Benchmark"), show="headings", height=6)
        self.tree.heading("Metric", text="평가지표 명")
        self.tree.heading("Portfolio", text="나의 ML 포트폴리오")
        self.tree.heading("Benchmark", text="S&P 500 (벤치마크)")
        
        self.tree.column("Metric", width=200, anchor="center")
        self.tree.column("Portfolio", width=180, anchor="center")
        self.tree.column("Benchmark", width=180, anchor="center")
        self.tree.pack(fill="both", expand=True)
        
        # 4. 자산 결과 요약 텍스트 박스
        self.result_text = tk.Text(self.root, height=4, bg="#f0f0f0", relief="flat", font=("Malgun Gothic", 10))
        self.result_text.pack(fill="x", padx=20, pady=10)
        self.result_text.insert("1.0", "결과 요약창: 백테스팅이 완료되면 이곳에 자산 가치가 표기됩니다.")
        self.result_text.config(state="disabled")

    def start_pipeline_thread(self):
        """대용량 데이터 다운로드 및 연산 시 화면이 멈추는 것을 방지하기 위해 쓰레드로 백그라운드 구동"""
        self.run_btn.config(state="disabled")
        self.status_label.config(text="⏳ S&P 500 장기 데이터 다운로드 및 가공 중... 잠시만 기다려주세요.", foreground="orange")
        
        # 테이블 초기화
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        threading.Thread(target=self.run_trading_pipeline, daemon=True).start()

    def run_trading_pipeline(self):
        try:
            # UI에서 입력받은 값 정제
            capital_val = float(self.capital_entry.get())
            rebalance_period = int(self.rebalance_var.get())
            alpha = self.config['ridge_alpha']
            start_date = self.config['start_date']
            end_date = self.config['end_date']
            
            # 1단계 연산
            raw_df = fetch_raw_data(None, start_date, end_date)
            self.root.after(0, lambda: self.status_label.config(text="⏳ 5팩터 추출 및 머신러닝 모델 학습 중..."))
            X, y = engineer_features(raw_df)
            
            # 2단계 연산
            model = train_and_update_model(X, y, alpha=alpha)
            predictions = predict_returns(model, X)
            
            # 3단계 연산 (UI의 리밸런싱 주기 적용)
            self.root.after(0, lambda: self.status_label.config(text="⏳ 자산 배분 조절 및 리스크 시뮬레이션 중..."))
            weights_history = allocate_assets(predictions, X, y, rebalance_period=rebalance_period)
            
            # 4단계 연산 (UI의 시작 금액 적용)
            summary_table, final_bal, spy_bal = run_backtest(y, weights_history, initial_capital=capital_val)
            
            # UI 화면 갱신 (Main Thread에서 실행되도록 after 처리)
            self.root.after(0, self.update_gui_results, summary_table, capital_val, final_bal, spy_bal)
            
        except Exception as e:
            self.root.after(0, self.handle_error, str(e))

    def update_gui_results(self, summary_table, initial, final, spy):
        # 표(Treeview)에 데이터 삽입
        for metric, row in summary_table.iterrows():
            self.tree.insert("", "end", values=(metric, row['나의 ML 포트폴리오'], row['S&P 500 (벤치마크)']))
            
        # 요약 텍스트 갱신
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        report_str = (
            f"💰 초기 투자 금액: {initial:,.0f} 달러\n"
            f"🏆 나의 ML 포트폴리오 최종 자산 가치: {final:,.2f} 달러\n"
            f"📉 S&P 500 매수 후 보유 시 최종 자산 가치: {spy:,.2f} 달러"
        )
        self.result_text.insert("1.0", report_str)
        self.result_text.config(state="disabled")
        
        self.status_label.config(text="✅ 백테스팅 연산 및 결과 매핑이 성공적으로 완료되었습니다!", foreground="green")
        self.run_btn.config(state="normal")

    def handle_error(self, err_msg):
        self.status_label.config(text="❌ 연산 오류 발생. 입력을 확인해 주세요.", foreground="red")
        messagebox.showerror("오류 발생", f"파이프라인 구동 중 에러가 발생했습니다:\n{err_msg}")
        self.run_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingGuiApp(root)
    root.mainloop()