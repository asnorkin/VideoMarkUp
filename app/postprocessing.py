import os
import os.path as osp
from shutil import copy, make_archive, rmtree

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import vtools as vt
from utils import (
    choose_detects,
    choose_labels,
    choose_video,
    config_page,
    get_arguments,
    load_sequences,
    title,
    Stage,
)


def postprocessing(args):
    title(stage=Stage.POSTPROCESSING)

    # Preprocess
    video_path = choose_video(args.data_dir)
    detects_path = choose_detects(args.data_dir)
    labels_path = choose_labels(args.output_dir)

    load_bar = st.progress(0)
    load_button = st.button("Start processing video")
    if load_button:
        process_data(labels_path, detects_path, video_path, args.output_dir, pbar=load_bar)
        st.text("Video successfully postprocessed.")


def process_data(labels_file, detects_file, video_file, output_dir, pbar=None):
    ext = "png"
    match_name = osp.splitext(osp.basename(video_file))[0]

    def image_file(track_id, frame_id):
        return f"track_{track_id}_frame_{frame_id}.{ext}"

    sequences = load_sequences(labels_file)

    # Generate labels df
    labels = []
    for track_id, track_sequences in sequences.items():
        for first, last, number in track_sequences.segments:
            for frame_id in range(first, last + 1):
                labels.append({
                    "track_id": track_id,
                    "rel_frame": frame_id,
                    "number": number,
                })
    labels_df = pd.DataFrame(labels, dtype=int)

    # Extract tracks meta
    detects = pd.read_csv(detects_file, low_memory=False)
    track_ids = np.unique(detects[~np.isnan(detects['track_id'])]['track_id']).astype(int)
    tracks = {track_id: detects[detects['track_id'] == track_id] for track_id in track_ids}

    frame_ids, rel_paths = [], []

    # Fill labels and absolute frames mapping
    frame_sets = {track_id: set() for track_id in track_ids}
    labels = {track_id: {} for track_id in track_ids}
    for i, row in labels_df.iterrows():
        track_id = row.track_id
        frame_no = tracks[track_id].iloc[row.rel_frame]["frame_id"]
        frame_sets[track_id].add(frame_no)
        labels[track_id][frame_no] = row.number

        # Add absolute frame_id and rel_path to
        frame_ids.append(frame_no)
        rel_paths.append(f"{match_name}/{image_file(track_id, frame_no)}")

    # Update labels_df
    labels_df["frame_id"] = frame_ids
    labels_df["rel_path"] = rel_paths
    labels_df.drop(columns=["rel_frame"], inplace=True)

    match_name = osp.splitext(osp.basename(video_file))[0]
    match_dir = os.path.join(output_dir, match_name)

    images_dir = os.path.join(match_dir, match_name)
    os.makedirs(images_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Can't open video"
    frame_no, last_frame = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        for track_id in track_ids:
            if frame_no in frame_sets[track_id]:
                bbox = vt.bbox_from_df(tracks[track_id], frame_no)
                _, orig_bbox = vt.get_bbox(frame, bbox, return_original_bbox=True)
                save_filename = osp.join(images_dir, image_file(track_id, frame_no))
                cv2.imwrite(save_filename, orig_bbox)

        frame_no += 1
        if pbar:
            pbar.progress(frame_no / last_frame)
        if frame_no == last_frame:
            break

    labels_df.to_csv(osp.join(match_dir, osp.splitext(osp.basename(labels_file))[0] + ".csv"), index=False)
    make_archive(os.path.join(output_dir, match_name), "zip", match_dir)
    rmtree(match_dir)


def main():
    config_page()
    postprocessing(get_arguments())


if __name__ == "__main__":
    main()
