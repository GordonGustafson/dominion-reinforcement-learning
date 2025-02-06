import strategies
import play

from sklearn.linear_model import LinearRegression

from chooser import Chooser


def train_scikit_learn_linear_model():
    random_choosers = [Chooser(strategies.random_strategy),
                       Chooser(strategies.random_strategy)]
    games_df, _ = play.play_n_games(["random_chooser_1", "random_chooser_2"], random_choosers, n=20)

    print(games_df)
    X = games_df.drop(columns=["reward"])
    y = games_df["reward"]
    model = LinearRegression().fit(X, y)
    print(model.coef_)
    print(model.intercept_)

    trained_chooser_func = strategies.scikit_learn_max_state_score_strategy(model)
    trained_chooser_funcs = [trained_chooser_func] * 2
    trained_choosers = [Chooser(f) for f in trained_chooser_funcs]

    trained_games_df, _ = play.play_n_games(["linear_model_1", "linear_model_2"], trained_choosers, n=2)

