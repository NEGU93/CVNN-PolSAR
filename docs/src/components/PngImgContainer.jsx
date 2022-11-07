import React from "react";
import { useSelector } from "react-redux";
import { getDataSetImgLink } from "../redux/slices/imagesSlice";
import { PNG_FILES_NAMES } from "../constants/constants";
import "../styles/ImgConatiner.scss";

const PngImgContainer = () => {
  const imgDataSet = getDataSetImgLink(
    useSelector((state) => state.imagesReducer.dataSet)
  );
  // console.log(imgDataSet);

  return imgDataSet !== "" ? (
    <div className="pngImgContainer">
      {PNG_FILES_NAMES.map((file, key) => (
        <div className="imageContainer" key={key}>
          <img
            className="pngImg"
            src={`/assets/${imgDataSet}/${file}`}
            alt=""
          />
        </div>
      )
    )}
    </div>
  ) : (
    <h3 className="defaultText">" Select a Dataset to begin "</h3>
  );
};

export default PngImgContainer;
