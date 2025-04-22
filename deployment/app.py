import streamlit as st
import prediction
import eda

st.set_page_config(
    page_title = 'Recycling Sorting App',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

st.sidebar.markdown("## ♻️ Welcome to the Recycling Sorting App")
st.sidebar.markdown(
    "<p style='font-size:12px;'>Use this app to classify and sort garbage for recycling purpose.</p>", 
    unsafe_allow_html=True
)
page = st.sidebar.selectbox('📊 **Navigate using below drop-down:** ', ('Explore Sample Data', 'Classify an Image'))
st.sidebar.markdown("---")
st.sidebar.write('Created by ***Galuh Alifani***')
st.sidebar.expander("See Source & Credits", expanded=False).markdown('- Data Source: [Kaggle: Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)\n - Yang, Mindy, Thung, Gary. (2016) [Classification of Trash for Recyclability Status](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf): Stanford University')

if page == 'Explore Sample Data':
    eda.run()
else:
    prediction.run()