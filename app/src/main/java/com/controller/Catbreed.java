package com.controller;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.databind.util.JSONPObject;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@RestController
@ResponseBody
//@RequestMapping("")
public class Catbreed {

    @PostMapping("/catbreed/uploading")
    public String upload(){
        JSONObject res=new JSONObject();
        res.put("breed","cat");
        res.put("gender","male");
        res.put("age","2");
        return res.toJSONString();
    }
}
