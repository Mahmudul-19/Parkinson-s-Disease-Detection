import streamlit as st

def app():
    col1, col2, col3 = st.columns([0.5,2,0.5])
    with col2:
        st.header("*Frequently Asked Question*")
    
    st.subheader(":red[What is Parkinson's disease?]")
    st.markdown("Parkinson disease (PD) is a brain condition that causes problems with movement, mental health, sleep, pain and other health issues.")
    
    st.subheader(":green[What is the cure of Parkinson's disease?]")
    st.markdown("PD gets worse over time. There is no cure, but therapies and medicines can reduce symptoms.")
    
    st.subheader(":violet[What are the symptoms of Parkinson's disease?]")
    st.markdown("Parkinson's disease symptoms can be different for everyone. Early symptoms may be mild and go unnoticed. Symptoms often begin on one side of the body and usually remain worse on that side, even after symptoms begin to affect the limbs on both sides. Parkinson's symptoms may include:\n1. Rhythmic shaking called tremor \n2. Rigid muscles \n3. Slowed movement known as bradykinesia\n4. Impaired posture and balance \n5. Loss of automatic movements \n6. Speech changes \n7. Writing changes ")
    
    st.subheader(":orange[What is the main cause of Parkinson's disease?]")
    st. markdown("The cause of Parkinson's essentially remains unknown. However, theories involving oxidative damage, environmental toxins, genetic factors and accelerated aging have been discussed as potential causes for the disease.")
    
    st.subheader(":red[How many attributes are used in the dataset to train the model?]")
    st.markdown("There are 22 attributes in the dataset that retrives from the voice of patients as a raw dataset. The attributes are Fundamental Frequency, Highest Frequency, Lowest frequency, Jitter Percentage, Absulate Jitter, MDVP: RAP, MDVP: PPQ, Jitter: DDP, MDVP: Shimmer, MDVP: Shimmer (dB), Shimmer: APQ3, Shimmer: APQ5, MDVP: APQ, Shimmer: DDA, NHR, HNR, RPDE, DFA, Spread1, Spread2, D2, PPE")