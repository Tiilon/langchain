from langchain_community.document_loaders import UnstructuredPowerPointLoader, UnstructuredExcelLoader, Docx2txtLoader
from scripts import llm

## PowerPoint
loader = UnstructuredPowerPointLoader("7. document_loaders/data/ml_course.pptx", mode="elements")

docs = loader.load()

# doc = docs[0]
# doc.metadata
# doc.page_content

ppt_data = {}
for doc in docs:
    page = doc.metadata["page_number"]
    ppt_data[page] = ppt_data.get(page, "")  + "\n\n" + doc.page_content

# print(ppt_data)

context = ""
for page, content in ppt_data.items():
    context += f"### Slide {page}:\n\n{content.strip()}\n\n\n"

# print(context)

question ="""
For each PowerPoint slide provided above, write a 2-minute script that effectively conveys the key points.
Ensure a smooth flow between slides, maintaining a clear and engaging narrative.
"""

response = llm.ask_llm(context, question)

# print(response)
with open("data/ppt_script.md", "w") as f:
    f.write(response)

############################################################################

### Excel
loader = UnstructuredExcelLoader("data/sample.xlsx",  mode="elements")
docs = loader.load()

len(docs)

doc = docs[0]
doc.metadata

doc.page_content

context = doc.metadata['text_as_html']

context

question = "Return this data in Markdown format."
response = llm.ask_llm(context, question)
print(response)

question = "Return all entris in the table where Gender is 'F'. Format the response in Markdown. Do not write preambles and explanation."
response = llm.ask_llm(context, question)
print(response)

question = "Return all entris in the table where Gender is 'male'. Format the response in Markdown. Do not write preambles and explanation."
response = llm.ask_llm(context, question)
print(response)

###########################################################
### Word Document
loader = Docx2txtLoader("data/job_description.docx")

docs = loader.load()

context = docs[0].page_content

# print(context)

question ="""
My name is Aaditya, and I am a recent graduate from IIT with a focus on Natural Language Processing and Machine Learning.
I am applying for a Data Scientist position at SpiceJet.
Please write a concise job application email for me in short, removing any placeholders, including references to job boards or sources.
"""
response = llm.ask_llm(context, question)
print(response)