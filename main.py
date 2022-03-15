from bga import BGA



if __name__ == "__main__":
    bga = BGA((0, 0), 0, chrom_l=[0, 0])
    a, b, c = bga.run() # start algorithm
    bga.plot(a, b, c) # show result and plot