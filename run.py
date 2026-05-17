import arxiv
import google.generativeai as genai
from google.api_core import exceptions
from datetime import datetime, timedelta, timezone
import os
import json
import time
import re
import random
import requests
import hashlib
import base64

# ================= 配置类 =================
class Config:
    BASE_DIR = os.path.dirname(__file__)
    HISTORY_FILE = os.path.join(BASE_DIR, 'analyzed_papers.json')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'daily_reports')
    IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
    
    WECHAT_WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=04902edb-0cb3-4680-a991-d010691aa083"
    
    # arXiv 搜索配置
    ARXIV_QUERY = (
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
    ARXIV_DAYS_THRESHOLD = 7
    
    # Gemini 提示词
    PROMPT_PHASE_1 = """
    作为一名电商平台的资深广告/推荐算法专家，请仔细阅读附件中的完整 PDF 论文。
    特别注意并结合论文中的**模型架构图（Architecture Diagram）**、流程图以及实验对比表格进行初步解读。
    
    论文基本信息：
    标题：{title}
    作者：{authors}
    链接：{entry_id}
    
    请针对“电商环境下的生成式广告与推荐”这一背景，结合你从 PDF 文本和框架图中获取的信息，按照以下格式输出第一阶段解读：
    
    ### 1. 🎯 核心一句话总结
    (请用通俗易懂的语言，一句话概括这篇论文试图解决电商广告/推荐场景中的什么问题，提出了什么方法)

    ### 2. 💡 核心创新点
    (详细列出论文的2-3个核心创新点，说明它与现有技术相比，突破了什么瓶颈？)

    ### 3. 🖼️ 核心算法与技术路径
    (非常详细地说明它的模型架构和算法原理。它是如何结合生成式模型（如 LLM/Diffusion 等）的？请描述你在论文中看到的关键模块、公式逻辑和数据流向。)
    
    ### 4. 📊 实验结论简述
    (简要说明实验是在什么数据集上验证的，对比了哪些基线，以及该方法带来了多少核心指标的提升。如果有代表性的实验结果对比数据，请提取并使用精简的 Markdown 表格展示。)

    ### 5. 📚 算法工程师阅读建议
    (强烈推荐阅读 / 选读 / 略读，并给出简短理由)
    """
    
    PROMPT_PHASE_2 = """
    非常好的解读。作为在工业界做电商推荐/广告算法的一线工程师，我们需要用更加严谨、挑剔的眼光来审视这篇学术论文的真实落地价值。
    
    请你作为“审稿人+工程落地负责人”，带着质疑的态度，再次深入论文细节，回答以下两个关键问题（如果论文中**完全没有提及**相关数据，请明确指出“**论文未提及**”，并给出你基于经验的推测）：

    ### 6. ⏱️ 在线推理耗时 (RT/Latency) 与工程落地分析
    - 大家都知道生成式模型（特别是 LLM/Diffusion）在线上预估时极其耗时。这篇论文是否公布了真实的在线推理耗时数据（Latency）？
    - 他们有没有提出针对线上部署的工程优化手段（如：大模型离线蒸馏、特征缓存 KV Cache、端云协同、双塔双层架构解耦等）？如果没有，你认为落地的最大瓶颈是什么？

    ### 7. 🧩 消融实验真实收益剖析
    - 仔细看他们的消融实验（Ablation Study）。生成式模块（LLM/Diffusion）带来的绝对提升到底有多少？
    - 这种提升是来自于“生成式模型真正懂了用户”，还是仅仅因为“大模型参数量多、加了更多特征”？
    """

    @classmethod
    def init_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.IMAGES_DIR, exist_ok=True)


# ================= 工具类 =================
class WeChatNotifier:
    @staticmethod
    def send_markdown(content):
        data = {"msgtype": "markdown", "markdown": {"content": content}}
        try:
            res = requests.post(Config.WECHAT_WEBHOOK_URL, json=data)
            return res.json()
        except Exception as e:
            print(f"❌ 发送微信消息失败: {e}")
            return None

    @staticmethod
    def send_image(image_path):
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            md5_hash = hashlib.md5(image_data).hexdigest()
            data = {"msgtype": "image", "image": {"base64": base64_data, "md5": md5_hash}}
            res = requests.post(Config.WECHAT_WEBHOOK_URL, json=data)
            return res.json()
        except Exception as e:
            print(f"❌ 发送微信图片失败: {e}")
            return None


class HistoryManager:
    def __init__(self):
        self.history_file = Config.HISTORY_FILE
        self.analyzed_set = self._load()

    def _load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def save(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.analyzed_set), f, indent=4)

    def is_analyzed(self, entry_id):
        return entry_id in self.analyzed_set

    def mark_analyzed(self, entry_id):
        self.analyzed_set.add(entry_id)
        self.save()


class ArxivFetcher:
    def __init__(self):
        self.client = arxiv.Client(page_size=20, delay_seconds=10, num_retries=10)
        self.search = arxiv.Search(
            query=Config.ARXIV_QUERY,
            max_results=40,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

    def fetch_recent_papers(self):
        # 增加随机休眠，避免 GitHub Actions 在整点同时发起请求导致 arXiv 接口 429 限流
        sleep_time = random.randint(30, 120)
        print(f"💤 为避免触发 arXiv 限流，随机休眠 {sleep_time} 秒...")
        time.sleep(sleep_time)

        time_threshold = datetime.now(timezone.utc) - timedelta(days=Config.ARXIV_DAYS_THRESHOLD)
        recent_papers = []
        max_search_retries = 5
        all_results = []

        for attempt in range(max_search_retries):
            try:
                print(f"👉 正在向 arXiv 发送搜索请求 (尝试 {attempt + 1}/{max_search_retries})...")
                time.sleep(attempt * 5)
                all_results = list(self.client.results(self.search))
                break
            except Exception as e:
                if "HTTP 429" in str(e) or attempt == max_search_retries - 1:
                    print(f"⚠️ arXiv 接口限流或报错: {e}")
                    if attempt < max_search_retries - 1:
                        wait_time = 60 * (attempt + 1) + random.randint(10, 30)
                        print(f"💤 休息 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print("❌ 搜索失败，跳过本次执行。")
                        return []
                else:
                    raise e

        print(f"👉 [调试] arXiv 接口初步搜索到了 {len(all_results)} 篇论文")
        if all_results:
            print(f"👉 [调试] 最新的论文发布时间为: {all_results[0].published}")
            print(f"👉 [调试] 我们设定的时间阈值是: {time_threshold}")

        for result in all_results:
            if result.published >= time_threshold:
                recent_papers.append(result)

        return recent_papers


class GeminiAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("请设置 GEMINI_API_KEY 环境变量，例如：export GEMINI_API_KEY='你的密钥'")
        
        genai.configure(api_key=api_key, transport="rest")
        self._check_models()
        self.model = genai.GenerativeModel('gemini-3-flash-preview')

    def _check_models(self):
        print("👉 [调试] 正在查询当前 API Key 支持的模型列表...")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"  - 可用模型: {m.name}")
        except Exception as e:
            print(f"查询模型列表失败，请检查网络或代理: {e}")

    def _download_pdf(self, paper, pdf_filename):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"📥 正在从 arXiv 下载论文 PDF (尝试 {attempt + 1}/{max_retries})...")
                paper.download_pdf(dirpath=".", filename=pdf_filename)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ 网络超时或下载失败，3秒后重试: {e}")
                time.sleep(3)
        return False

    def _upload_to_gemini(self, pdf_filename):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"📤 正在上传 PDF 至 Gemini (尝试 {attempt + 1}/{max_retries})...")
                uploaded_file = genai.upload_file(path=pdf_filename)
                return uploaded_file
            except exceptions.ResourceExhausted:
                print("🚨 触发配额限制，强制休眠 60 秒...")
                time.sleep(60)
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ 网络超时或上传失败，3秒后重试: {e}")
                time.sleep(3)
        return None

    def _wait_for_processing(self, uploaded_file):
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            raise ValueError("Gemini PDF 文件解析失败")
        return uploaded_file

    def analyze_paper(self, paper):
        pdf_filename = f"temp_{paper.get_short_id()}.pdf"
        uploaded_file = None
        full_response = ""

        try:
            self._download_pdf(paper, pdf_filename)
            uploaded_file = self._upload_to_gemini(pdf_filename)
            uploaded_file = self._wait_for_processing(uploaded_file)

            time.sleep(10) # 执行分析前，先睡几秒，保证 RPM 平稳
            
            prompt1 = Config.PROMPT_PHASE_1.format(
                title=paper.title,
                authors=[author.name for author in paper.authors],
                entry_id=paper.entry_id
            )

            print(f"\n{'='*50}\n【论文】{paper.title}\n【链接】{paper.entry_id}\n{'-'*50}")
            chat = self.model.start_chat()

            # 阶段一
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

            # 阶段二
            print("\n\n[阶段二] 正在以工业界视角进行深度反思与拷问...\n")
            try:
                response2 = chat.send_message(Config.PROMPT_PHASE_2, stream=True)
                full_response2 = ""
                for chunk in response2:
                    print(chunk.text, end="", flush=True)
                    full_response2 += chunk.text
            except exceptions.ResourceExhausted:
                print("🚨 触发配额限制，强制休眠 60 秒...")
                time.sleep(60)
                raise

            print(f"\n{'='*50}\n")
            full_response = full_response1 + "\n\n---\n\n" + full_response2

        finally:
            if uploaded_file:
                try:
                    genai.delete_file(uploaded_file.name)
                except Exception:
                    pass
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)

        return full_response


class ReportGenerator:
    def __init__(self):
        self.today_str = datetime.now().strftime('%Y-%m-%d')
        self.report_filename = os.path.join(Config.OUTPUT_DIR, f"daily_report_{self.today_str}.md")

    def append_paper_analysis(self, paper, analysis_text):
        with open(self.report_filename, 'a', encoding='utf-8') as f_report:
            if os.path.getsize(self.report_filename) == 0:
                f_report.write(f"# 🤖 电商广告/推荐前沿论文日报 ({self.today_str})\n\n")
            
            f_report.write(f"## 📄 [{paper.title}]({paper.entry_id})\n")
            f_report.write(f"**Authors:** {', '.join([author.name for author in paper.authors])}\n\n")
            f_report.write(f"{analysis_text}\n\n")
            f_report.write("---\n\n")


# ================= 主流程 =================
def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始执行论文分析任务...")
    Config.init_dirs()
    
    history_manager = HistoryManager()
    fetcher = ArxivFetcher()
    
    papers = fetcher.fetch_recent_papers()
    if not papers:
        print("今天没有相关领域的新论文发布。")
        return

    new_papers = [p for p in papers if not history_manager.is_analyzed(p.entry_id)]
    if not new_papers:
        print(f"今天找到了 {len(papers)} 篇近期论文，但都已经分析过了，跳过处理。")
        return

    print(f"其中 {len(new_papers)} 篇为全新论文，准备开始分析...")
    
    analyzer = GeminiAnalyzer()
    report_gen = ReportGenerator()

    for paper in new_papers:
        print(f"\n正在准备分析论文: {paper.title} ...")
        try:
            analysis_text = analyzer.analyze_paper(paper)
            
            # 提取一句话总结用于微信推送
            summary_match = re.search(r'### 1\..*?核心一句话总结\n(.*?)(?=\n###|$)', analysis_text, re.S)
            one_line_summary = summary_match.group(1).strip() if summary_match else "点击查看详细解读"

            wechat_msg = (
                f"## 📄 论文速递: {paper.title}\n\n"
                f"> **核心总结**: {one_line_summary}\n\n"
                f"🔗 [查看原文]({paper.entry_id}) | [查看详细报告](https://github.com/NobodyWHU/auto-paper/tree/main/daily_reports)"
            )
            
            print(f"📤 正在发送微信推送: {paper.title}...")
            WeChatNotifier.send_markdown(wechat_msg)

            report_gen.append_paper_analysis(paper, analysis_text)
            history_manager.mark_analyzed(paper.entry_id)
            
        except exceptions.ResourceExhausted:
            print("🚨 触发配额限制，强制休眠 60 秒...")
            time.sleep(60)
        except Exception as e:
            print(f"❌ 处理 {paper.title} 时发生错误: {e}")
        
        time.sleep(3)

if __name__ == "__main__":
    main()
