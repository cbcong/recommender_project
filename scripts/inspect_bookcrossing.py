import os
import pandas as pd


def main():
    # 项目根目录：.../recommender_project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "bookcrossing")

    ratings_path = os.path.join(data_dir, "BX-Book-Ratings.csv")
    users_path = os.path.join(data_dir, "BX-Users.csv")
    books_path = os.path.join(data_dir, "BX-Books.csv")

    print("=== 路径检查 ===")
    print("ratings_path:", ratings_path)
    print("users_path:  ", users_path)
    print("books_path:  ", books_path)

    # Book-Crossing 原始文件是分号分隔、latin-1 编码，可能有脏行，用 on_bad_lines='skip'
    print("\n=== 读取 CSV 中（可能会稍微有点慢） ===")
    ratings = pd.read_csv(
        ratings_path,
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip'
    )
    users = pd.read_csv(
        users_path,
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip'
    )
    books = pd.read_csv(
        books_path,
        sep=';',
        encoding='latin-1',
        on_bad_lines='skip'
    )

    print("\n=== 各表前几行预览 ===")
    print("\n[BX-Book-Ratings.csv] head:")
    print(ratings.head())
    print("\n[BX-Users.csv] head:")
    print(users.head())
    print("\n[BX-Books.csv] head:")
    print(books.head())

    # 列名一般是：User-ID, ISBN, Book-Rating
    print("\n=== 列名检查 ===")
    print("ratings columns:", list(ratings.columns))
    print("users columns:  ", list(users.columns))
    print("books columns:  ", list(books.columns))

    # 基本规模
    n_interactions = len(ratings)
    n_users = ratings["User-ID"].nunique()
    n_items = ratings["ISBN"].nunique()
    print("\n=== 基本统计 ===")
    print(f"交互数（ratings 行数）: {n_interactions}")
    print(f"唯一用户数: {n_users}")
    print(f"唯一图书数: {n_items}")

    # 评分分布（0~10，0 代表隐式，1~10 为显式评分）
    print("\n=== 评分分布 (Book-Rating) ===")
    print(ratings["Book-Rating"].value_counts().sort_index())

    # 年龄分布（可能有很多 NaN 和离谱值）
    if "Age" in users.columns:
        age = users["Age"]
        print("\n=== 年龄统计 (原始) ===")
        print("年龄非空数量:", age.notna().sum())
        print("年龄描述性统计:")
        print(age.describe())

        # 只看合理范围 [5, 100] 的年龄
        age_clean = age[(age >= 5) & (age <= 100)]
        print("\n=== 年龄统计 (5~100 过滤后) ===")
        print("过滤后非空数量:", age_clean.notna().sum())
        print(age_clean.describe())

    # 地理位置分布：看一下 Location 前几种
    if "Location" in users.columns:
        print("\n=== Location 示例（TOP 10） ===")
        print(users["Location"].value_counts().head(10))

    print("\n=== 检查结束 ===")


if __name__ == "__main__":
    main()
