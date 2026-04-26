import arxiv
import google.generativeai as genai
from google.api_core import exceptions
from datetime import datetime, timedelta, timezone
import os
import json
import time
import re
import requests
import hashlib
import base64

# ================= 配置文件路径 =================
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'analyzed_papers.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'daily_reports')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

# 企业微信 Webhook 地址
WECHAT_WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=04902edb-0cb3-4680-a991-d010691aa083"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

def send_wechat_markdown(content):
    """发送 Markdown 消息到企业微信"""
    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": content
        }
    }
    try:
        res = requests.post(WECHAT_WEBHOOK_URL, json=data)
        return res.json()
    except Exception as e:
        print(f"❌ 发送微信消息失败: {e}")
        return None

def send_wechat_image(image_path):
    """发送图片消息到企业微信"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            
        # 计算图片的 base64 和 md5
        base64_data = base64.b64encode(image_data).decode('utf-8')
        md5_hash = hashlib.md5(image_data).hexdigest()
        
        data = {
            "msgtype": "image",
            "image": {
                "base64": base64_data,
                "md5": md5_hash
            }
        }
        res = requests.post(WECHAT_WEBHOOK_URL, json=data)
        return res.json()
    except Exception as e:
        print(f"❌ 发送微信图片失败: {e}")
        return None

def load_history():
    """加载已经分析过的论文 ID 记录"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_history(history_set):
    """保存已经分析过的论文 ID 记录"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(history_set), f, indent=4)

# 1. 配置 Gemini (建议通过环境变量读取 API KEY 以保证安全)
# 请在终端运行: export GEMINI_API_KEY="your_api_key_here"
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("请设置 GEMINI_API_KEY 环境变量，例如：export GEMINI_API_KEY='你的密钥'")

genai.configure(api_key=api_key)

# 增加一段调试代码：打印当前 API Key 到底支持哪些模型
print("👉 [调试] 正在查询当前 API Key 支持的模型列表...")
available_models = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"  - 可用模型: {m.name}")
except Exception as e:
    print(f"查询模型列表失败，请检查网络或代理: {e}")

# 尝试使用最轻量级的 lite 模型，它的免费配额通常是最宽松的
model = genai.GenerativeModel('gemini-3-flash-preview') 




def get_daily_papers():
    # 2. 构建针对“电商广告算法工程师（生成式推荐）”的专属检索词
    # 扩大覆盖面，包含大语言模型、生成式搜索、电商推荐、计算广告等核心领域
    query = (
        '('
        'all:"generative recommendation" OR '
        'all:"LLM recommendation" OR '
        'all:"generative advertising" OR '
        'all:"computational advertising" OR '
        'all:"CTR prediction" OR '
        'all:"conversion rate prediction" OR '
        'all:"e-commerce recommendation" OR '
        'all:"sponsored search" OR '
        '(all:"large language model" AND all:"recommendation") OR '
        '(all:"large language model" AND all:"advertising") OR '
        '(all:"diffusion model" AND all:"recommendation")'
        ')'
    )
    
    # 构造客户端
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=100, # 将最大拉取数量从 20 扩大到 100，避免漏掉近期的论文
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # 获取过去 7 天的论文 (放宽时间限制以便测试，原为 2 天)
    time_threshold = datetime.now(timezone.utc) - timedelta(days=7)
    
    recent_papers = []
    
    # 增加调试信息：看看 API 到底返回了多少数据
    all_results = list(client.results(search))
    print(f"👉 [调试] arXiv 接口初步搜索到了 {len(all_results)} 篇论文")
    if all_results:
        print(f"👉 [调试] 最新的论文发布时间为: {all_results[0].published}")
        print(f"👉 [调试] 我们设定的时间阈值是: {time_threshold}")
        
    for result in all_results:
        if result.published >= time_threshold:
            recent_papers.append(result)
            
    return recent_papers

def analyze_papers_with_gemini(papers):
    if not papers:
        print("今天没有相关领域的新论文发布。")
        return

    # 加载已处理的论文记录
    analyzed_history = load_history()
    
    # 过滤掉已经分析过的论文
    new_papers = [p for p in papers if p.entry_id not in analyzed_history]
    
    if not new_papers:
        print(f"今天找到了 {len(papers)} 篇近期论文，但都已经分析过了，跳过处理。")
        return
        
    print(f"其中 {len(new_papers)} 篇为全新论文，准备开始分析...")
    
    
    # 创建今天的 Markdown 报告文件
    today_str = datetime.now().strftime('%Y-%m-%d')
    report_filename = os.path.join(OUTPUT_DIR, f"daily_report_{today_str}.md")
    
    with open(report_filename, 'a', encoding='utf-8') as f_report:
        # 如果文件刚创建，写入标题
        if os.path.getsize(report_filename) == 0:
            f_report.write(f"# 🤖 电商广告/推荐前沿论文日报 ({today_str})\n\n")
            
        for result in new_papers:
            print(f"\n正在准备分析论文: {result.title} ...")
            pdf_filename = f"temp_{result.get_short_id()}.pdf"
            uploaded_file = None
            
            try:
                # 1. 下载 PDF 到本地 (增加网络重试机制)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"📥 正在从 arXiv 下载论文 PDF (尝试 {attempt + 1}/{max_retries})...")
                        result.download_pdf(dirpath=".", filename=pdf_filename)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        print(f"⚠️ 网络超时或下载失败，3秒后重试: {e}")
                        time.sleep(3)
                
                # 2. 上传 PDF 给 Gemini 进行视觉与全文解析 (增加网络重试机制)
                for attempt in range(max_retries):
                    try:
                        print(f"📤 正在上传 PDF 至 Gemini (尝试 {attempt + 1}/{max_retries})...")
                        uploaded_file = genai.upload_file(path=pdf_filename)
                        break
                    except exceptions.ResourceExhausted:
                        print("🚨 触发配额限制，强制休眠 60 秒...")
                        time.sleep(60)
                        if attempt == max_retries - 1:
                            raise
                        continue
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        print(f"⚠️ 网络超时或上传失败，3秒后重试: {e}")
                        time.sleep(3)
                
                # 等待文件在云端处理完成
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(2)
                    uploaded_file = genai.get_file(uploaded_file.name)
                    
                if uploaded_file.state.name == "FAILED":
                    raise ValueError("Gemini PDF 文件解析失败")
                
                # 执行分析前，先睡几秒，保证 RPM 平稳
                time.sleep(10)
                prompt1 = f"""
                作为一名电商平台的资深广告/推荐算法专家，请仔细阅读附件中的完整 PDF 论文。
                特别注意并结合论文中的**模型架构图（Architecture Diagram）**、流程图以及实验对比表格进行初步解读。
                
                论文基本信息：
                标题：{result.title}
                作者：{[author.name for author in result.authors]}
                链接：{result.entry_id}
                
                请针对“电商环境下的生成式广告与推荐”这一背景，结合你从 PDF 文本和框架图中获取的信息，按照以下格式输出第一阶段解读：
                
                ### 1. 🎯 核心一句话总结
                (请用通俗易懂的语言，一句话概括这篇论文试图解决电商广告/推荐场景中的什么问题，提出了什么方法)

                ### 2. 🖼️ 核心框架与技术路径 (重点结合论文里的图表)
                (详细说明它的模型架构是怎样的？请描述你在框架图/流程图中看到的关键模块和数据流向。它是如何结合生成式模型（如 LLM/Diffusion 等）的？)
                
                ### 3. 📚 算法工程师阅读建议
                (强烈推荐阅读 / 选读 / 略读，并给出简短理由)

                ### 4. 📄 提取架构图
                (请你自行判断这篇论文是否有一张非常关键、有助于理解其核心思路的模型架构图或主流程图。如果有，且你认为对读者很有帮助，请找出它位于 PDF 的第几页，并在回复的最后一行单独输出代码：【架构图页码：X】（X为纯数字）。如果该论文没有重点架构图，或者你认为图表辅助意义不大，请输出：【架构图页码：无】)
                """
                
                # 第二轮的反思/追问 Prompt
                prompt2 = """
                非常好的解读。作为在工业界做电商推荐/广告算法的一线工程师，我们需要用更加严谨、挑剔的眼光来审视这篇学术论文的真实落地价值。
                
                请你作为“审稿人+工程落地负责人”，带着质疑的态度，再次深入论文细节，回答以下三个关键问题（如果论文中**完全没有提及**相关数据，请明确指出“**论文未提及**”，并给出你基于经验的推测）：
                
                ### 5. 📊 实验基线与数据集拷问
                - 论文使用的对比基线（Baselines）是最新的工业界强基线（如 DIN, DIEN, SIM, DCNv2 等）吗，还是拿弱模型当沙包？
                - 实验是在公开小数据集（如 MovieLens）上跑的，还是在真实的工业级（百亿/千亿级规模）数据集上验证过？

                ### 6. ⏱️ 在线推理耗时 (RT/Latency) 与工程落地分析
                - 大家都知道生成式模型（特别是 LLM/Diffusion）在线上预估时极其耗时。这篇论文是否公布了真实的在线推理耗时数据（Latency）？
                - 他们有没有提出针对线上部署的工程优化手段（如：大模型离线蒸馏、特征缓存 KV Cache、端云协同、双塔双层架构解耦等）？如果没有，你认为落地的最大瓶颈是什么？

                ### 7. 🧩 消融实验真实收益剖析
                - 仔细看他们的消融实验（Ablation Study）。生成式模块（LLM/Diffusion）带来的 CTR/CVR 绝对提升（AUC 或 LogLoss）到底有多少？
                - 这种提升是来自于“生成式模型真正懂了用户”，还是仅仅因为“大模型参数量多、加了更多文本特征”？
                """
                
                print(f"\n{'='*50}\n【论文】{result.title}\n【链接】{result.entry_id}\n{'-'*50}")
                
                # 开启多轮对话模式
                chat = model.start_chat()
                
                # 第 1 轮对话：基础解析
                print("\n[阶段一] 正在进行基础解析与架构图提取...\n")
                try:
                    response1 = chat.send_message([uploaded_file, prompt1], stream=True)
                    full_response1 = ""
                    for chunk in response1:
                        print(chunk.text, end="", flush=True)
                        full_response1 += chunk.text
                except exceptions.ResourceExhausted:
                    print("🚨 触发配额限制，强制休眠 60 秒...")
                    time.sleep(60)
                    raise
                
                # 第 2 轮对话：深度反思与拷问
                print("\n\n[阶段二] 正在以工业界视角进行深度反思与拷问...\n")
                try:
                    response2 = chat.send_message(prompt2, stream=True)
                    full_response2 = ""
                    for chunk in response2:
                        print(chunk.text, end="", flush=True)
                        full_response2 += chunk.text
                except exceptions.ResourceExhausted:
                    print("🚨 触发配额限制，强制休眠 60 秒...")
                    time.sleep(60)
                    raise
                    
                print(f"\n{'='*50}\n")
                
                # 合并两轮的回答
                full_response = full_response1 + "\n\n---\n\n" + full_response2
                
                # 尝试从回复中解析出“架构图页码”
                image_md = ""
                img_filepath = ""
                match = re.search(r'【架构图页码：(\d+)】', full_response)
                if match:
                    page_num = int(match.group(1))
                    print(f"👉 [系统] Gemini 指示核心架构图在第 {page_num} 页，正在尝试截取...")
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(pdf_filename)
                        if 0 < page_num <= len(doc):
                            # PyMuPDF 的页码是从 0 开始的
                            page = doc.load_page(page_num - 1)
                            # 渲染当前页为高清图片 (dpi=150)
                            pix = page.get_pixmap(dpi=150)
                            img_filename = f"{result.get_short_id()}_arch.png"
                            img_filepath = os.path.join(IMAGES_DIR, img_filename)
                            pix.save(img_filepath)
                            print(f"✅ 成功截取第 {page_num} 页作为框架图！")
                            image_md = f"\n\n**核心架构图 (Page {page_num}):**\n![Architecture Diagram](images/{img_filename})\n\n"
                    except ImportError:
                        print("⚠️ 提示：缺少 PyMuPDF 库，无法截取图片。请在终端运行：pip install PyMuPDF")
                    except Exception as e:
                        print(f"⚠️ 截取图片失败: {e}")

                # 提取一句话总结用于微信推送 (增强鲁棒性)
                summary_match = re.search(r'### 1\..*?核心一句话总结\n(.*?)(?=\n###|$)', full_response, re.S)
                one_line_summary = summary_match.group(1).strip() if summary_match else "点击查看详细解读"

                # 发送微信推送
                wechat_msg = f"## 📄 论文速递: {result.title}\n\n" \
                             f"> **核心总结**: {one_line_summary}\n\n" \
                             f"🔗 [查看原文]({result.entry_id}) | [查看详细报告](https://github.com/NobodyWHU/auto-paper/tree/main/daily_reports)"
                
                print(f"📤 正在发送微信推送: {result.title}...")
                send_wechat_markdown(wechat_msg)
                if img_filepath and os.path.exists(img_filepath):
                    send_wechat_image(img_filepath)

                # 将分析结果保存到 Markdown 文件中
                f_report.write(f"## 📄 [{result.title}]({result.entry_id})\n")
                f_report.write(f"**Authors:** {', '.join([author.name for author in result.authors])}\n\n")
                if image_md:
                    f_report.write(image_md)
                f_report.write(f"{full_response}\n\n")
                f_report.write("---\n\n")
                
                # 标记该论文已分析并保存
                analyzed_history.add(result.entry_id)
                save_history(analyzed_history)
                
            except exceptions.ResourceExhausted:
                print("🚨 触发配额限制，强制休眠 60 秒...")
                time.sleep(60)
            except Exception as e:
                print(f"❌ 处理 {result.title} 时发生错误: {e}")
            finally:
                # 清理云端和本地的临时 PDF 文件，避免堆积
                if uploaded_file:
                    try:
                        genai.delete_file(uploaded_file.name)
                    except:
                        pass
                if os.path.exists(pdf_filename):
                    os.remove(pdf_filename)
            
            # 即使成功，也保持 3 秒以上的间隔
            time.sleep(3)

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始获取最新 arXiv 论文...")
    papers = get_daily_papers()
    print(f"共找到 {len(papers)} 篇近期相关论文。开始调用 Gemini 分析...")
    analyze_papers_with_gemini(papers)
