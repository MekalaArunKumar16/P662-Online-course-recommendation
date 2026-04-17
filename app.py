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
import gdown
import os

# ----------------------------
# Helper: Download + Load pickle
# ----------------------------
@st.cache_resource
def load_pickle(file_id, filename):
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

    with open(filename, "rb") as f:
        return pickle.load(f)

# ----------------------------
# Load Data
# ----------------------------
@st.cache_resource
def load_data():
    df = load_pickle("1enXm6lGEYBRIv8tXpcnmrTYjUDtYwjgY", "data.pkl")
    course_sim = load_pickle("1c7y7WoNfyZ4OCTLMJ3RT8GdIc8tSi5E8", "course_sim.pkl")
    item_sim = load_pickle("14--VTq7owuE1a6QwK26_xJs2fuDZnyA9", "item_sim.pkl")
    user_item = load_pickle("1Lm_tPczxvysZs2pu64vHaZ6eKLwEk9F7", "user_item.pkl")

    return df, course_sim, item_sim, user_item

# ----------------------------
# Load once
# ----------------------------
with st.spinner("🔄 Loading data (first time only)..."):
    df, course_sim_df, item_sim_df, user_item_matrix = load_data()

# Convert index to int list (fixes matching issue)
valid_user_ids = set(map(int, user_item_matrix.index.tolist()))

# ----------------------------
# Recommendation Functions
# ----------------------------
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

    # ❌ New user → Popular
    if int(user_id) not in valid_user_ids:
        st.warning("⚠️ User not found → Using Popular Recommendation System")
        return get_popular_courses(n)

    # ✅ Existing user → Hybrid
    st.success("✅ Existing user detected → Using Hybrid Recommendation System")

    user_ratings = user_item_matrix.loc[user_id]
    rated_courses = user_ratings[user_ratings > 0]

    if rated_courses.empty:
        st.info("📊 No history → Showing popular courses")
        return get_popular_courses(n)

    scores = {}

    for course, rating in rated_courses.items():
        if course in item_sim_df.columns:
            similar_items = item_sim_df[course]

            for sim_course, sim_score in similar_items.items():
                if sim_course not in rated_courses.index:
                    scores[sim_course] = scores.get(sim_course, 0) + sim_score * rating

    if not scores:
        return get_popular_courses(n)

    rec_df = pd.DataFrame(scores.items(), columns=["course_id", "score"])
    rec_df = rec_df.merge(df, on="course_id")
    rec_df = rec_df.sort_values(by="score", ascending=False)

    return rec_df[['course_id', 'course_name', 'instructor', 'rating']].drop_duplicates().head(n)

# ----------------------------
# UI DESIGN
# ----------------------------
st.title("🎓 Online Course Recommender")
st.markdown("### 🚀 Hybrid ML-based Personalized Recommendations")

# Sidebar
st.sidebar.header("🔧 Controls")

min_id = int(min(valid_user_ids))
max_id = int(max(valid_user_ids))

st.sidebar.markdown(f"**Valid User IDs:** {min_id} → {max_id}")

user_id = st.sidebar.number_input(
    "Enter User ID (try outside range for new user)",
    value=min_id,
    step=1
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
