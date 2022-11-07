import React, { Fragment } from "react";
import { useSelector } from "react-redux";
import { getInputImgLink } from "../redux/slices/imagesSlice";
import { HTML_FILES_NAMES } from "../constants/constants";
import "../styles/ImgConatiner.scss";

const HtmlImgComponent = (props) => {
  const imgPath = getInputImgLink(useSelector((state) => state.imagesReducer));
  
  return imgPath !== "" ? (
    <Fragment>
      {HTML_FILES_NAMES.map((file, key) => {
        if (!(file === "lines-plot.html" && imgPath.includes("test")) && (file != "per-class-bar.html" || !(imgPath.includes("loss") || imgPath.includes("average_accuracy") )))
          return <div className="htmlImgContainer" key={key}>
            <div className="imageContainer">
            <iframe
              className="htmlImg"
              id="inlineFrameExample"
              title="Inline Frame Example"
              width="100%"
              height="100%"
              frameBorder="0"
              src={`/assets/${imgPath}/${file}`}
            ></iframe>
          </div>
        </div>
      })}
    </Fragment>
  ) : (
    ""
  );
};

export default HtmlImgComponent;
