
import streamlit as st
import google.generativeai as genai
from exa_py import Exa
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os

# --- Blog Generation Functions ---

def summarize_serp_results(serp_results, gemini_api_key, max_to_summarize=5):
    """Summarize the content of each SERP/blog result using Gemini."""
    summaries = []
    for i, result in enumerate(serp_results[:max_to_summarize]):
        url = getattr(result, 'url', None)
        title = getattr(result, 'title', None)
        # Try all possible content fields
        content = (
            getattr(result, 'content', None)
            or getattr(result, 'extract', None)
            or getattr(result, 'description', None)
            or getattr(result, 'snippet', None)
            or ''
        )
        if not content:
            # If no content, just summarize the title and URL
            summaries.append({
                'title': title,
                'url': url,
                'summary': f'(No content available to summarize. Title: {title}, URL: {url})'
            })
            continue
        prompt = f"""
        Summarize the following blog content for competitive analysis. Identify the main topics, structure, and any unique value or product recommendations. Keep the summary concise (5-7 bullet points):\n\nTitle: {title}\nURL: {url}\nContent:\n{content}\n"""
        summary = generate_text_with_exception_handling(prompt, gemini_api_key)
        summaries.append({
            'title': title,
            'url': url,
            'summary': summary.strip() if summary else f'(Failed to summarize. Title: {title}, URL: {url})'
        })
    return summaries

def generate_blog_post(input_blog_keywords, input_type, input_tone, input_language, metaphor_api_key, gemini_api_key, num_serp_results):
    serp_results = None
    try:
        serp_results = metaphor_search_articles(input_blog_keywords, metaphor_api_key, num_serp_results)
    except Exception as err:
        st.error(f"❌ Failed to retrieve search results for {input_blog_keywords}: {err}")
    if serp_results:
        # Summarize top SERP/blogs for analysis
        summaries = summarize_serp_results(serp_results, gemini_api_key, max_to_summarize=min(5, num_serp_results))
        # Optionally display summaries to user
        with st.expander("Top Competitor Blog Summaries (SERP Analysis)", expanded=False):
            for s in summaries:
                st.markdown(f"**[{s['title']}]({s['url']})**")
                st.markdown(s['summary'])
                st.markdown('---')
        # Use summaries in the prompt for better blog generation
        summaries_text = '\n\n'.join([f"Title: {s['title']}\nSummary: {s['summary']}" for s in summaries])
        prompt = f"""
        You are an experienced SEO strategist and creative content writer who specializes in crafting {input_type} blog posts in {input_language}. Your blog posts are designed to rank highly in search results while deeply engaging readers with a professional yet personable tone.\n\n        ### Task:\n        Write a comprehensive, engaging, and SEO-optimized blog post on the topic below. The blog should:\n        - Be structured for readability with clear headings, subheadings, and bullet points.\n        - Include actionable insights, real-world examples, and personal anecdotes to make the content relatable and practical.\n        - Be written in a {input_tone} tone that balances professionalism with a conversational style.\n\n        ### Requirements:\n        1. **SEO Optimization**:\n           - Use the provided keywords naturally and strategically throughout the content.\n           - Incorporate semantic keywords and related terms to enhance search engine visibility.\n           - Align the content with Google's E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) guidelines.\n\n        2. **Content Structure**:\n           - Start with a compelling introduction that hooks the reader and outlines the blog's value.\n           - Organize the content with logical headings and subheadings.\n           - Use bullet points, numbered lists, and short paragraphs for readability.\n\n        3. **Engagement and Value**:\n           - Provide actionable tips, real-world examples, and personal anecdotes.\n           - Include at least one engaging call-to-action (CTA) to encourage reader interaction.\n\n        4. **FAQs Section**:\n           - Include 5 FAQs derived from “People also ask” queries and related search suggestions.\n           - Provide thoughtful, well-researched answers to each question.\n\n        5. **Visual and Multimedia Suggestions**:\n           - Recommend where to include images, infographics, or videos to enhance the content's appeal.\n\n        6. **SEO Metadata**:\n           - Append the following metadata after the main blog content:\n             - A **Blog Title** that is catchy and includes the primary keyword.\n             - A **Meta Description** summarizing the blog post in under 160 characters.\n             - A **URL Slug** that is short, descriptive, and formatted in lowercase with hyphens.\n             - A list of **Hashtags** relevant to the content.\n\n        ### Blog Details:\n        - **Title**: {input_blog_keywords}\n        - **Keywords**: {input_blog_keywords}\n        - **SERP Competitor Summaries**:\n        {summaries_text}\n\n        Now, craft an exceptional blog post that stands out in search results and delivers maximum value to readers.\n        """
        return generate_text_with_exception_handling(prompt, gemini_api_key)
    return None

def metaphor_search_articles(query, api_key, num_results):
    if not api_key:
        raise ValueError("Metaphor API Key is missing!")
    metaphor = Exa(api_key)
    try:
        search_response = metaphor.search_and_contents(query, use_autoprompt=True, num_results=num_results)
        return search_response.results
    except Exception as err:
        st.error(f"Failed in metaphor.search_and_contents: {err}")
        return None

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config={"max_output_tokens": 8192})
        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text
    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
        return None

st.set_page_config(page_title="AI Affiliate Blog Writer", layout="wide")
st.markdown("""
    <style>
    ::-webkit-scrollbar-track { background: #e1ebf9; }
    ::-webkit-scrollbar-thumb { background-color: #90CAF9; border-radius: 10px; border: 3px solid #e1ebf9; }
    ::-webkit-scrollbar-thumb:hover { background: #64B5F6; }
    ::-webkit-scrollbar { width: 16px; }
    div.stButton > button:first-child {
        background: #1565C0; color: white; border: none; padding: 12px 24px; border-radius: 8px;
        text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 10px 2px;
        cursor: pointer; transition: background-color 0.3s ease; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<style>header {visibility: hidden;}</style>', unsafe_allow_html=True)
st.markdown('<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>', unsafe_allow_html=True)

st.title("✍️ AI Affiliate Blog Writer")
st.markdown("Create high-quality Amazon affiliate blog content with real-time research. 🚀")

with st.expander("API Configuration 🔑", expanded=False):
    st.markdown('''If the default API keys are unavailable or exceed their limits, you can provide your own API keys below.<br>
    <a href="https://metaphor.systems/" target="_blank">Get Metaphor API Key</a><br>
    <a href="https://aistudio.google.com/app/apikey" target="_blank">Get Gemini API Key</a>
    ''', unsafe_allow_html=True)
    user_metaphor_api_key = st.text_input("Metaphor API Key", type="password", help="Paste your Metaphor API Key here if you have one.")
    user_gemini_api_key = st.text_input("Gemini API Key", type="password", help="Paste your Gemini API Key here if you have one.")


with st.expander("**PRO-TIP** - Read the instructions below. 📝", expanded=True):
    col1, col2, col3 = st.columns([5, 5, 5])
    with col1:
        input_blog_keywords = st.text_input('**🔑 Enter main keywords of your blog!** (Blog Title Or Content Topic)', help="The main topic or title for your blog.")
        blog_type = st.selectbox('📝 Blog Post Type', options=['General', 'How-to Guides', 'Listicles', 'Job Posts', 'Cheat Sheets', 'Customize'], index=0)
        if blog_type == 'Customize':
            blog_type = st.text_input("Enter your custom blog type", help="Provide a custom blog type if you chose 'Customize'.")
        num_serp_results = st.slider('🔎 Number of SERP/blog results to analyze', min_value=10, max_value=100, value=10, step=1, help="How many top-ranking blogs to analyze for this keyword.")
    with col2:
        input_blog_tone = st.selectbox('🎨 Blog Tone', options=['General', 'Professional', 'Casual', 'Customize'], index=0)
        if input_blog_tone == 'Customize':
            input_blog_tone = st.text_input("Enter your custom blog tone", help="Provide a custom blog tone if you chose 'Customize'.")
    with col3:
        input_blog_language = st.selectbox('🌐 Language', options=['English', 'Vietnamese', 'Chinese', 'Hindi', 'Spanish', 'Customize'], index=0)
        if input_blog_language == 'Customize':
            input_blog_language = st.text_input("Enter your custom language", help="Provide a custom language if you chose 'Customize'.")



def main():
    if st.button('**Write Blog Post ✍️**'):
        with st.spinner('Generating your blog post...'):
            if not input_blog_keywords:
                st.error('**🫣 Provide Inputs to generate Blog Post. Keywords are required!**')
            else:
                metaphor_api_key = user_metaphor_api_key or os.getenv('METAPHOR_API_KEY')
                gemini_api_key = user_gemini_api_key or os.getenv('GEMINI_API_KEY')
                if not metaphor_api_key:
                    st.error("❌ Metaphor API Key is not available! Please provide your API key in the API Configuration section.")
                elif not gemini_api_key:
                    st.error("❌ Gemini API Key is not available! Please provide your API key in the API Configuration section.")
                else:
                    try:
                        blog_post = generate_blog_post(
                            input_blog_keywords,
                            blog_type,
                            input_blog_tone,
                            input_blog_language,
                            metaphor_api_key,
                            gemini_api_key,
                            num_serp_results
                        )
                        if blog_post:
                            st.subheader('**👩🧕🔬 Your Final Blog Post!**')
                            st.write(blog_post)
                        else:
                            st.error("💥 Failed to generate blog post. Please try again!")
                    except Exception as e:
                        if "quota exceeded" in str(e).lower():
                            st.error("❌ API limit exceeded! Please provide your own API key in the API Configuration section.")
                        else:
                            st.error(f"💥 An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


# --- Blog Generation Functions ---
def generate_blog_post(input_blog_keywords, input_type, input_tone, input_language, metaphor_api_key, gemini_api_key):
    serp_results = None
    try:
        serp_results = metaphor_search_articles(input_blog_keywords, metaphor_api_key)
    except Exception as err:
        st.error(f"❌ Failed to retrieve search results for {input_blog_keywords}: {err}")
    if serp_results:
        prompt = f"""
        You are an experienced SEO strategist and creative content writer who specializes in crafting {input_type} blog posts in {input_language}. Your blog posts are designed to rank highly in search results while deeply engaging readers with a professional yet personable tone.\n\n        ### Task:\n        Write a comprehensive, engaging, and SEO-optimized blog post on the topic below. The blog should:\n        - Be structured for readability with clear headings, subheadings, and bullet points.\n        - Include actionable insights, real-world examples, and personal anecdotes to make the content relatable and practical.\n        - Be written in a {input_tone} tone that balances professionalism with a conversational style.\n\n        ### Requirements:\n        1. **SEO Optimization**:\n           - Use the provided keywords naturally and strategically throughout the content.\n           - Incorporate semantic keywords and related terms to enhance search engine visibility.\n           - Align the content with Google's E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) guidelines.\n\n        2. **Content Structure**:\n           - Start with a compelling introduction that hooks the reader and outlines the blog's value.\n           - Organize the content with logical headings and subheadings.\n           - Use bullet points, numbered lists, and short paragraphs for readability.\n\n        3. **Engagement and Value**:\n           - Provide actionable tips, real-world examples, and personal anecdotes.\n           - Include at least one engaging call-to-action (CTA) to encourage reader interaction.\n\n        4. **FAQs Section**:\n           - Include 5 FAQs derived from “People also ask” queries and related search suggestions.\n           - Provide thoughtful, well-researched answers to each question.\n\n        5. **Visual and Multimedia Suggestions**:\n           - Recommend where to include images, infographics, or videos to enhance the content's appeal.\n\n        6. **SEO Metadata**:\n           - Append the following metadata after the main blog content:\n             - A **Blog Title** that is catchy and includes the primary keyword.\n             - A **Meta Description** summarizing the blog post in under 160 characters.\n             - A **URL Slug** that is short, descriptive, and formatted in lowercase with hyphens.\n             - A list of **Hashtags** relevant to the content.\n\n        ### Blog Details:\n        - **Title**: {input_blog_keywords}\n        - **Keywords**: {input_blog_keywords}\n        - **Google SERP Results**: {serp_results}\n\n        Now, craft an exceptional blog post that stands out in search results and delivers maximum value to readers.\n        """
        return generate_text_with_exception_handling(prompt, gemini_api_key)
    return None

def metaphor_search_articles(query, api_key):
    if not api_key:
        raise ValueError("Metaphor API Key is missing!")
    metaphor = Exa(api_key)
    try:
        search_response = metaphor.search_and_contents(query, use_autoprompt=True, num_results=5)
        return search_response.results
    except Exception as err:
        st.error(f"Failed in metaphor.search_and_contents: {err}")
        return None

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text_with_exception_handling(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config={"max_output_tokens": 8192})
        convo = model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text
    except Exception as e:
        st.exception(f"An unexpected error occurred: {e}")
        return None
