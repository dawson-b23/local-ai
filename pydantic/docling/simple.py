from docling.document_converter import DocumentConverter 

source = input("enter file name: ")
try:
    converter = DocumentConverter()
    doc = converter.convert(source).document 
    with open("output.md", "w") as file:
        file.write(doc.export_to_markdown())
except Exception as e:
    print(f"** ** ** ERROR ** ** **, {str(e)}")

#print(doc.export_to_markdown())
