from bga import BGA



if __name__ == "__main__":
    bga = BGA((4, 6), 0, chrom_l=[2, 4])
    a,b = bga.run()
    bga.plot(a, b)