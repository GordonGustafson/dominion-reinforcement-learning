import strategies
import play

from sklearn.linear_model import LinearRegression

def train_scikit_learn_linear_model():
    random_chooser_funcs = [strategies.random_strategy,
                            strategies.random_strategy]
    games_df, _ = play.play_n_games(["random_chooser_1", "random_chooser_2"], random_chooser_funcs, n=20)

    print(games_df)
    X = games_df.drop(columns=["reward"])
    y = games_df["reward"]
    model = LinearRegression().fit(X, y)
    print(model.coef_)
    print(model.intercept_)

    trained_chooser_func = strategies.scikit_learn_state_scoring_model_strategy(model)
    trained_chooser_funcs = [trained_chooser_func] * 2

    trained_games_df, _ = play.play_n_games(["linear_model_1", "linear_model_2"], trained_chooser_funcs, n=2)

