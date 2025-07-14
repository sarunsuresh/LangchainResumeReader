import streamlit as st
from Engine import load_resume_text, build_qa_chain

st.set_page_config(page_title="Resume Q&A Bot", page_icon="📄")
st.title(" Resume Q&A Bot")

# Upload resume PDF
uploaded_pdf = st.file_uploader("Upload your resume (PDF only)", type="pdf")

if uploaded_pdf:
    with open("resume.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("✅ Resume uploaded successfully!")

    # Load text and build chain
    text = load_resume_text("resume.pdf")
    qa = build_qa_chain(text)

    # User input
    st.subheader("Ask a question about your resume:")
    question = st.text_input("Your question")

    if question:
        with st.spinner("Thinking..."):
            answer = qa.run(question)
        st.success("✅ Answer:")
        st.write(answer)
