import matplotlib.pyplot as plt
from detr_rdd.configs import COLORS

def get_output_figure(pil_img, results, threshold):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    for result in results:
        score = result["score"]
        label = result["label"]
        box = list(result["box"].values())
        if score > threshold:
            c = COLORS[hash(label) % len(COLORS)]
            ax.add_patch(
                plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color=c, linewidth=3)
            )
            text = f"{label}: {score:0.2f}"
            ax.text(box[0], box[1], text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")

    return plt.gcf()