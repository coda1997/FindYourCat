import {wxuuid} from 'util'

Page({
  onLoad() {
    this.ctx = wx.createCameraContext()
  },
  takePhoto() {
    this.ctx.takePhoto({
      quality: 'high',
      success: (res) => {
        this.setData({
          src: res.tempImagePath
        })
      }
    })
  },
  error(e) {
    console.log(e.detail)
  },
  uploadFile() {
    wx.uploadFile({
      filePath: this.src,
      name: `${wxuuid()}`,
      url: 'url',
      formData: formData,
      header: header,
      timeout: 0,
      success: (result) => {},
      fail: (res) => {},
      complete: (res) => {},
    })
  },
})