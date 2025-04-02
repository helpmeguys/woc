import streamlit as st

st.title("File Upload Example (including .py files)")

uploaded_file = st.file_uploader(
    "Choose a file to upload",
    type=["csv", "txt", "pdf", "py"]  # Added "py" for Python scripts
)

if uploaded_file is not None:
    # Read file as bytes
    bytes_data = uploaded_file.getvalue()
    
    st.write("**Filename:**", uploaded_file.name)
    st.write("**File type:**", uploaded_file.type)
    st.write("**File size:**", len(bytes_data), "bytes")
    
    # Optionally, display the contents if it's a text-based file
    try:
        string_data = uploaded_file.read().decode("utf-8")
        st.text_area("File Contents", string_data, height=300)
    except Exception as e:
        st.error(f"Could not decode file contents: {e}")
