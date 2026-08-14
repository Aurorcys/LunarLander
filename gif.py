import imageio

reader = imageio.get_reader("videos/lunar_landing_rank4_h264.mp4")
fps = reader.get_meta_data()['fps']

writer = imageio.get_writer("lunar_landing_rank4.gif", fps=fps)
for frame in reader:
    writer.append_data(frame)
writer.close()