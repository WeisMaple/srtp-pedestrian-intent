"""Local classroom presentation UI; inference uses the released video pipeline."""
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from modern_pipeline.bootstrap import configure, PROJECT_ROOT

OUTPUT = Path(os.environ.get('SRTP_PRESENTATION_OUTPUT', 'D:/SRTP_presentation_20261029'))

class DemoApp:
    def __init__(self, root):
        self.root=root; root.title('行人过街意图分析 · 课堂演示'); root.geometry('1280x820')
        self.events=queue.Queue(); self.frames=queue.Queue(maxsize=1)
        self.stop=threading.Event(); self.worker=None; self.closing=False
        self.mode=tk.StringVar(value='本地道路视频'); self.source=tk.StringVar()
        self.predict=tk.BooleanVar(value=True); self.save=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value='请选择道路视频。摄像头模式默认只显示骨架与跟踪。')
        self.controls=[]
        ttk.Label(root,text='行人过街意图分析',font=('Microsoft YaHei UI',22,'bold')).pack(anchor='w',padx=24,pady=(18,6))
        ttk.Label(root,text='YOLO姿态估计 · 多人跟踪 · LSTM骨架预测 · ST-GCN分类').pack(anchor='w',padx=24)
        bar=ttk.Frame(root);bar.pack(fill='x',padx=24,pady=14)
        mode=ttk.Combobox(bar,textvariable=self.mode,values=['本地道路视频','摄像头 / 视频流'],state='readonly',width=20)
        mode.pack(side='left');mode.bind('<<ComboboxSelected>>',self.change_mode);self.controls.append(mode)
        entry=ttk.Entry(bar,textvariable=self.source);entry.pack(side='left',fill='x',expand=True,padx=10);self.controls.append(entry)
        browse=ttk.Button(bar,text='选择视频',command=self.browse);browse.pack(side='left');self.controls.append(browse)
        opts=ttk.Frame(root);opts.pack(fill='x',padx=24)
        for label,var in [('启用过街预测（道路场景）',self.predict),('保存分析视频与运行记录',self.save)]:
            b=ttk.Checkbutton(opts,text=label,variable=var);b.pack(side='left',padx=(0,20));self.controls.append(b)
        self.start_button=ttk.Button(opts,text='开始分析',command=self.start);self.start_button.pack(side='left',padx=8)
        self.stop_button=ttk.Button(opts,text='停止',command=self.stop.set,state='disabled');self.stop_button.pack(side='left')
        ttk.Button(opts,text='打开结果文件夹',command=self.open_output).pack(side='right')
        ttk.Label(root,text='实时流可填写摄像头编号（如 0）或 RTSP / HTTP 地址。教室场景仅用于骨架跟踪演示。',wraplength=1200).pack(anchor='w',padx=24,pady=10)
        views=ttk.Frame(root);views.pack(fill='both',expand=True,padx=24)
        self.panels=[]
        for title in ['原始画面','分析画面']:
            box=ttk.LabelFrame(views,text=title);box.pack(side='left',fill='both',expand=True,padx=4)
            panel=tk.Label(box,bg='#16202b',fg='white',text='等待输入',font=('Microsoft YaHei UI',16));panel.pack(fill='both',expand=True);self.panels.append(panel)
        ttk.Label(root,textvariable=self.status,wraplength=1200,font=('Microsoft YaHei UI',11)).pack(fill='x',padx=24,pady=14)
        ttk.Label(root,text='分类：Other＝无关行人，Not crossing＝不过街，Crossing＝过街。预测分数不是安全保证。').pack(anchor='w',padx=24,pady=(0,14))
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(60,self.poll)

    def change_mode(self,event=None):
        live=self.mode.get()!='本地道路视频';self.predict.set(not live);self.source.set('0' if live else '')
        self.status.set('实时流默认只展示骨架与ID；网络流断开可能需数秒退出。' if live else '本地视频逐帧处理，不跳帧；处理速度与播放帧率分别显示。')

    def browse(self):
        p=filedialog.askopenfilename(filetypes=[('视频','*.mp4 *.avi *.mov *.mkv'),('所有文件','*.*')])
        if p:self.mode.set('本地道路视频');self.predict.set(True);self.source.set(p)

    def open_output(self):
        OUTPUT.mkdir(parents=True,exist_ok=True);os.startfile(str(OUTPUT))

    def start(self):
        if self.worker and self.worker.is_alive():return
        source=self.source.get().strip();live=self.mode.get()!='本地道路视频'
        if not source or (not live and not Path(source).is_file()):
            messagebox.showerror('输入无效','请选择存在的视频文件，或填写摄像头编号/视频流地址。');return
        if live and self.predict.get() and not messagebox.askyesno('道路视频流预测','该模型针对车载道路场景。教室场景不适合解释为过街意图；视频流采样时序也可能不同于训练数据。确认对道路视频流启用预测？'):return
        self.stop.clear();self.start_button.configure(state='disabled');self.stop_button.configure(state='normal')
        for c in self.controls:c.configure(state='disabled')
        self.status.set('正在加载模型…');self.worker=threading.Thread(target=self.run,args=(source,live,self.predict.get(),self.save.get()),daemon=True);self.worker.start()

    def run(self,source,live,predict,save):
        try:
            configure()
            from modern_pipeline.cli import arguments,run
            weights=PROJECT_ROOT/'weights/final';cfg=json.loads((weights/'model_config.json').read_text(encoding='utf-8'))
            output=OUTPUT/('session_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)+'.mp4')
            args=['--input',source if not live else 'live','--output',str(output),'--pose-model',str(weights/cfg['pose_model']),
                  '--stgcn',str(weights/cfg['stgcn']),'--lstm',str(weights/cfg['lstm']),'--normalization',cfg['normalization'],'--stats',str(weights/cfg['stats']),
                  '--behavior-device','cpu','--presentation']
            if not save:args+=['--no-video']
            def frame(original,annotated,index,people,ready,elapsed):
                import cv2
                images=[cv2.cvtColor(cv2.resize(im,(640,360)),cv2.COLOR_BGR2RGB) for im in [original,annotated]]
                try:self.frames.get_nowait()
                except queue.Empty:pass
                self.frames.put_nowait((images,f'已处理 {index+1} 帧  |  处理 {(index+1)/max(elapsed,.001):.1f} FPS  |  行人 {people}  |  分类输出 {ready}'))
            meta=run(arguments(args),on_frame=frame,stop_event=self.stop,live_source=(int(source) if source.isdigit() else source) if live else None,pose_only=not predict)
            self.events.put(('done',f"{'已停止' if meta['status']=='stopped' else '处理完成'} · {meta['frames']} 帧 · {meta['processing_fps']:.2f} FPS · 输入标称 {meta['input_fps']:.2f} FPS · 结果：{output if save else output.with_suffix('.metrics.json')}"))
        except Exception as exc:self.events.put(('error',f'{type(exc).__name__}: {exc}'))

    def poll(self):
        try:
            images,status=self.frames.get_nowait()
            from PIL import Image,ImageTk
            for panel,im in zip(self.panels,images):
                obj=Image.fromarray(im);obj.thumbnail((max(100,panel.winfo_width()),max(100,panel.winfo_height())))
                photo=ImageTk.PhotoImage(obj);panel.configure(image=photo,text='');panel.image=photo
            self.status.set(status)
        except queue.Empty:pass
        try:
            kind,text=self.events.get_nowait();self.status.set(text)
            self.start_button.configure(state='normal');self.stop_button.configure(state='disabled')
            for c in self.controls:c.configure(state='readonly' if isinstance(c,ttk.Combobox) else 'normal')
            if kind=='error' and not self.closing:messagebox.showerror('运行失败',text)
        except queue.Empty:pass
        if self.closing and not (self.worker and self.worker.is_alive()):self.root.destroy();return
        self.root.after(60,self.poll)

    def close(self):
        self.closing=True;self.stop.set();self.status.set('正在停止并保存结果…')

if __name__=='__main__':
    configure();root=tk.Tk();DemoApp(root);root.mainloop()
