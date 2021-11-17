import streamlit as st


def app():
    st.title('Welcome')
    st.markdown('A web dashboard created using Streamlit.')
    
    st.header('Project Description')
    st.markdown('This project is a Social Distancing & Human Action Recognition Detector implemented in Python with OpenCV and OpenPose. '
                'These tools are able to access live video streams from CCTV footage to automatically estimate interpersonal distance, '
                'detect crowd and abnormal human action. The goal is to help the community, including TARUC staff and students to ensure '
                'ensure social distancing protocol and safety in their workplace.')

    st.header('Scope of functionalities')
    st.subheader('Social Distancing Detector')
    st.caption('• Detect humans in the frame with yolov3')
    st.caption('• Estimate number of people who violate social distance')
    st.caption('• Create trend chart of social distancing violations')
    st.caption('• Send real-time notification if crowd is detected')

    st.subheader('Human Action Recognition Detector')
    st.caption('• Detect humans in the frame with tensorflow openpose')
    st.caption('• Estimate the abnormal behaviour')
    st.caption('• Create trend chart of human actions')
    st.caption('• Send real-time notification if abnormal behaviour occurs')

    expander_faq = st.expander("More About Our Project")
    expander_faq.write("Hi there! If you have any questions about our project, or simply want to check out the source code, please visit our github repo: https://github.com/chinyew24/Smart-Crowd-Monitoring-System")
