# ----------------------------
# FIRST STREAMLIT COMMAND
# ----------------------------
import streamlit as st
st.set_page_config(page_title="Online Course Recommender", layout="wide")

# ----------------------------
# Imports
# ----------------------------
import pickle
import pandas as pd

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pickle.load(open('data.pkl', 'rb'))
    course_sim = pickle.load(open('course_sim.pkl', 'rb'))
    item_sim = pickle.load(open('item_sim.pkl', 'rb'))
    user_item = pickle.load(open('user_item.pkl', 'rb'))
    return df, course_sim, item_sim, user_item

df, course_sim_df, item_sim_df, user_item_matrix = load_data()

# ----------------------------
# Recommendation Functions
# ----------------------------
def content_based_recommendations(course_id, n=5):
    if course_id not in course_sim_df.columns:
        return pd.DataFrame()

    sim_scores = course_sim_df[course_id].sort_values(ascending=False)
    top_courses = sim_scores.iloc[1:n+1].index

    return df[df['course_id'].isin(top_courses)][
        ['course_id', 'course_name', 'instructor', 'rating']
    ].drop_duplicates()


def get_popular_courses(n=5):
    popular = df.groupby(['course_id', 'course_name', 'instructor']).agg({
        'enrollment_numbers': 'sum',
        'rating': 'mean'
    }).reset_index()

    popular = popular.sort_values(
        ['enrollment_numbers', 'rating'],
        ascending=False
    )

    return popular.head(n)[
        ['course_id', 'course_name', 'instructor', 'rating']
    ]


def hybrid_recommendations(user_id, n=5):

    # 🆕 Case 1: New user
    if user_id not in user_item_matrix.index:
        st.info("🆕 New user detected → Showing popular courses")
        return get_popular_courses(n)

    user_ratings = user_item_matrix.loc[user_id]
    rated_courses = user_ratings[user_ratings > 0].index.tolist()

    # 😶 Case 2: Existing user but no ratings
    if len(rated_courses) == 0:
        st.info("📊 No history found → Showing popular courses")
        return get_popular_courses(n)

    # 🤖 Case 3: Hybrid recommendations
    recs = []

    for course in rated_courses:
        if course in course_sim_df.columns:
            recs.append(content_based_recommendations(course, n))

    if recs:
        recs = pd.concat(recs).drop_duplicates()

        # Remove already seen courses
        recs = recs[
            ~recs['course_id'].isin(rated_courses)
        ]

        recs = recs.sort_values(by='rating', ascending=False)

        st.success("🤖 Personalized recommendations ready!")
        return recs.head(n)

    # fallback
    return get_popular_courses(n)


# ----------------------------
# UI DESIGN
# ----------------------------

# Header
st.title("🎓 Online Course Recommender")
st.markdown("### 🚀 Hybrid ML-based Personalized Recommendations")

# Sidebar
st.sidebar.header("🔧 Controls")

# 👉 Number input instead of dropdown
user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=int(user_item_matrix.index.min()),
    max_value=int(user_item_matrix.index.max()),
    value=int(user_item_matrix.index.min()),
    step=1
)

st.sidebar.caption(
    f"Valid IDs: {user_item_matrix.index.min()} to {user_item_matrix.index.max()}"
)

n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# ----------------------------
# Recommendation Trigger
# ----------------------------
if st.sidebar.button("✨ Get Recommendations"):

    with st.spinner("Generating recommendations..."):
        recommendations = hybrid_recommendations(user_id, n)

    st.markdown("---")
    st.subheader("📌 Recommended Courses")

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(3)

        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        padding:15px;
                        border-radius:12px;
                        background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
                        color:white;
                        margin-bottom:15px;
                        box-shadow:0 4px 15px rgba(0,0,0,0.3);
                    ">
                        <h4>📚 {row['course_name']}</h4>
                        <p><b>👨‍🏫 Instructor:</b> {row['instructor']}</p>
                        <p><b>⭐ Rating:</b> {round(row['rating'],2)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("💡 Built with Streamlit | Hybrid Recommendation System")