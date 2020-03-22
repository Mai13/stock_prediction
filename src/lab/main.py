from stock_optimizer import StockOptimizer


def main():

    stock_optimizer = StockOptimizer()
    stock_optimizer.run()


if __name__ == '__main__':

    try:
        main()
    except Exception as e:
        print(e)  # set a logger
