import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def show():
   
    sns.set_style("whitegrid")

    
    custom_palette = ['#5DADE2', '#58D68D', '#F5B041', '#E74C3C']

   
    plan_descriptions = {
        "Basic Plan": "A basic telecom plan offering minimal features, such as limited talk time, basic mobile data, and essential services.",
        "Standard Plan": "A mid-tier plan offering a balance between cost and features. Includes moderate talk time, more mobile data, and some additional services.",
        "Premium Plan": "A feature-rich plan with extensive talk time, large mobile data limits, high-speed internet, and value-added services.",
        "Ultimate Plan": "The top-tier plan with unlimited talk time, the highest mobile data, high-speed internet, and all premium services."
    }

  
    st.title("Plan Details and Analytics")

    st.header("Service Plans")
    for plan_name, description in plan_descriptions.items():
        st.subheader(plan_name)
        st.write(description)

  
    df = pd.read_csv('predicted_service_plans.csv')


    plan_counts = df['BestServiceName'].value_counts()

   
    st.header("User Distribution Across Plans")

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(plan_counts.index, plan_counts.values, color=custom_palette, edgecolor='black', linewidth=0.7)


    for bar in bars:
        bar.set_linewidth(1)
        bar.set_edgecolor('black')
        bar.set_capstyle('round')

    ax.set_facecolor('#f5f5f5')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Service Plans", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Users", fontsize=14, fontweight='bold')
    ax.set_title("Number of Users for Each Service Plan", fontsize=16, fontweight='bold')


    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    st.pyplot(fig)
    if st.button("Back to Admin Dashboard"):
      
       
        st.session_state['current_page'] = 'Admin'
            
        st.rerun()
