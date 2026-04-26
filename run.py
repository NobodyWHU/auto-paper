import arxiv
import google.generativeai as genai
from datetime import datetime, timedelta, timezone
import os
import json

# ================= 配置文件路径 =================
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'analyzed_papers.json')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'daily_reports')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
model = genai.GenerativeModel('gemini-2.5-flash-lite') 




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
            print(f"\n正在分析论文: {result.title} ...")
            prompt = f"""
        作为一名电商平台的资深广告/推荐算法专家，请分析以下来自 arXiv 的最新论文：
        
        标题：{result.title}
        作者：{[author.name for author in result.authors]}
        摘要：{result.summary}
        链接：{result.entry_id}
        
        请针对“电商环境下的生成式广告与推荐”这一背景，按照以下格式输出解读：
        
        ### 1. 🎯 核心一句话总结
        (请用通俗易懂的语言，一句话概括这篇论文试图解决电商广告/推荐场景中的什么问题，提出了什么方法)

        ### 2. 💡 核心创新点与生成式技术路径
        (它是如何结合生成式模型（如 LLM/Diffusion 等）的？相比传统判别式基线模型（如双塔/DeepFM/DIN 等 CTR 模型）有何优势？)

        ### 3. 🚀 工业界广告业务落地潜力评估
        (这篇论文的方法在真实的电商广告/搜索推荐业务中落地可行性如何？对点击率(CTR)/转化率(CVR)预估、创意生成、或者长尾物料分发有何帮助？可能面临的耗时/算力瓶颈是什么？)
        
        ### 4. 📚 算法工程师阅读建议
        (强烈推荐阅读 / 选读 / 略读，并给出简短理由)
        """
        
        try:
            print(f"\n{'='*50}\n【论文】{result.title}\n【链接】{result.entry_id}\n{'-'*50}")
            # 开启 stream=True，实现打字机效果，让你实时看到进度
            response = model.generate_content(prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
            print(f"\n{'='*50}\n")
            
            # 将分析结果保存到 Markdown 文件中
            f_report.write(f"## 📄 [{result.title}]({result.entry_id})\n")
            f_report.write(f"**Authors:** {', '.join([author.name for author in result.authors])}\n\n")
            f_report.write(f"{full_response}\n\n")
            f_report.write("---\n\n")
            
            # 标记该论文已分析并保存
            analyzed_history.add(result.entry_id)
            save_history(analyzed_history)
            
        except Exception as e:
            print(f"调用 Gemini API 解析 {result.title} 时发生错误: {e}")

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始获取最新 arXiv 论文...")
    papers = get_daily_papers()
    print(f"共找到 {len(papers)} 篇近期相关论文。开始调用 Gemini 分析...")
    analyze_papers_with_gemini(papers)
