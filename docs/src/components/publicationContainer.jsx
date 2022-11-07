import { useSelector } from "react-redux";
import { getDataSetImgLink } from "../redux/slices/imagesSlice";
import { PUBLICATION_FILES } from "../constants/constants";

const PublicationComponent = (props) => {
    const dataset = getDataSetImgLink(
        useSelector((state) => state.imagesReducer.dataSet)
      );
    return Boolean(dataset) ? ( <div>
        <h2>{PUBLICATION_FILES[dataset]["year"]}</h2>
        <h2>{PUBLICATION_FILES[dataset]["authors"]}</h2>
        <h2>{PUBLICATION_FILES[dataset]["title"]}</h2>
        <h2>{PUBLICATION_FILES[dataset]["published"]}</h2>
    </div>) : ("");
  };
  
  export default PublicationComponent;